import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import stu

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  use_stu: bool = False
  stu_num_eigh: int = 24
  stu_max_seq: int = 64
  stu_use_hankel_L: bool = False
  stu_units: int = 0
  # 'posthoc' = Run A: STU refines deter chain after the scan (legacy).
  # 'core'    = Run B: STU runs inside _core on a rolling action-history
  #             buffer kept in carry, and feeds into the GRU as a 4th input.
  stu_mode: str = 'posthoc'
  stu_random: bool = False
  stu_random_normalized: bool = False
  stu_neg_bank: bool = False
  stu_outscale: float = 0.0
  stu_shortcut: bool = False
  stu_use_obs: bool = False

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    self.act_space = act_space
    self.kw = kw
    # Total dim of the dict-concat'd action vector. DictConcat one-hots
    # discrete keys (classes per element of shape) and flattens continuous
    # keys (prod shape). We mirror that here so the STU buffer is sized
    # correctly at carry init time, before any action is seen.
    dim = 0
    for key in sorted(act_space.keys()):
      space = act_space[key]
      if space.discrete:
        classes = int(np.asarray(space.classes).flatten()[0])
        dim += int(np.prod(space.shape)) * classes
      else:
        dim += int(np.prod(space.shape))
    self._action_dim = dim

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    carry = dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32))
    if self.use_stu and self.stu_mode == 'core':
      carry['stu_buffer'] = jnp.zeros(
          [bsize, self.stu_max_seq, self._action_dim], f32)
      if self.stu_use_obs:
        carry['stu_obs_buffer'] = jnp.zeros(
            [bsize, self.stu_max_seq, self.stoch * self.classes], f32)
    return nn.cast(carry)

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    new_carry = jax.tree.map(lambda x: x[:, -1], entries)
    if self.use_stu and self.stu_mode == 'core':
      # The STU action buffer is not stored in replay (would inflate replay
      # storage by ~W× the action footprint per step). Instead, reset the
      # buffer to zeros at every replay-chunk boundary; the buffer then
      # rebuilds within the chunk as the scan progresses.
      bsize = new_carry['deter'].shape[0]
      new_carry['stu_buffer'] = nn.cast(jnp.zeros(
          [bsize, self.stu_max_seq, self._action_dim], f32))
      if self.stu_use_obs:
        new_carry['stu_obs_buffer'] = nn.cast(jnp.zeros(
            [bsize, self.stu_max_seq, self.stoch * self.classes], f32))
    return new_carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    out = jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)
    out = self._ensure_stu_buffer(out)
    return out

  def _ensure_stu_buffer(self, carry):
    """Add a zero-filled stu_buffer to carry if stu_mode='core' and the
    key is missing. Needed wherever a carry is reconstructed from
    `entries` (replay context, imagine starts) — we deliberately do NOT
    store the buffer in entries to avoid bloating replay storage, so the
    buffer must be re-materialized at chunk boundaries.
    """
    if not (self.use_stu and self.stu_mode == 'core'):
      return carry
    if 'stu_buffer' in carry:
      return carry
    bsize = jax.tree.leaves(carry)[0].shape[0]
    carry = {**carry, 'stu_buffer': nn.cast(jnp.zeros(
        [bsize, self.stu_max_seq, self._action_dim], f32))}
    if self.stu_use_obs and 'stu_obs_buffer' not in carry:
      carry['stu_obs_buffer'] = nn.cast(jnp.zeros(
          [bsize, self.stu_max_seq, self.stoch * self.classes], f32))
    return carry

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry = self._ensure_stu_buffer(carry)
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      if self.use_stu and self.stu_mode == 'posthoc':
        # Run A: causal STU refinement of the deter chain, applied as a
        # residual. We do NOT recompute the posterior here — the original
        # logit/stoch from _observe were used to update the GRU carry, so
        # they must remain canonical. STU only refines the deter that the
        # heads (decoder, reward, critic) consume. We also write refined
        # deter back into entries so the next replay segment's carry is
        # consistent with what the heads saw.
        refined = self._apply_stu(entries['deter'])
        entries = {**entries, 'deter': refined}
        feat = {**feat, 'deter': refined}
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    if self.use_stu and self.stu_mode == 'core':
      buffer = nn.mask(carry['stu_buffer'], ~reset)
      obs_buffer = None
      if self.stu_use_obs:
        obs_buffer = nn.mask(carry['stu_obs_buffer'], ~reset)
      buffer, obs_buffer, stu_ctx = self._stu_step(
          buffer, action, stoch, obs_buffer)
      deter = self._core(deter, stoch, action, stu_ctx)
    else:
      deter = self._core(deter, stoch, action)
      buffer = None
      obs_buffer = None
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    if buffer is not None:
      carry['stu_buffer'] = buffer
    if obs_buffer is not None:
      carry['stu_obs_buffer'] = obs_buffer
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    carry = self._ensure_stu_buffer(carry)
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      if self.use_stu and self.stu_mode == 'core':
        obs_buffer = carry.get('stu_obs_buffer', None)
        buffer, obs_buffer, stu_ctx = self._stu_step(
            carry['stu_buffer'], actemb, carry['stoch'], obs_buffer)
        deter = self._core(carry['deter'], carry['stoch'], actemb, stu_ctx)
      else:
        deter = self._core(carry['deter'], carry['stoch'], actemb)
        buffer = None
        obs_buffer = None
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      new_carry = dict(deter=deter, stoch=stoch)
      if buffer is not None:
        new_carry['stu_buffer'] = buffer
      if obs_buffer is not None:
        new_carry['stu_obs_buffer'] = obs_buffer
      carry = nn.cast(new_carry)
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # NOTE: in stu_mode='posthoc' (Run A) we intentionally do NOT
      # post-process the imagined deter chain — the next imagine carry
      # would come from unrefined _core, creating a train/imagine
      # asymmetry. In stu_mode='core' (Run B) STU is already integrated
      # inside _core via _stu_step, so the imagined feats are consistent
      # with the imagined carry by construction and no extra work is
      # needed here.
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action, stu_context=None):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    branches = [x0, x1, x2]
    if stu_context is not None:
      x3 = self.sub('dynin3', nn.Linear, self.hidden, **self.kw)(stu_context)
      x3 = nn.act(self.act)(self.sub('dynin3norm', nn.Norm, self.norm)(x3))
      branches.append(x3)
    x = jnp.concatenate(branches, -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _stu_step(self, buffer, action, stoch, obs_buffer=None):
    """Slide buffers by one and apply STUCore.

    buffer:     (B, W, action_dim) — action history.
    action:     (B, action_dim)    — current step's masked, dict-concat action.
    stoch:      (B, stoch, classes) — previous timestep's stoch from carry.
    obs_buffer: (B, W, stoch*classes) or None — observation (stoch) history.

    Returns:
      new_buffer:     updated action buffer
      new_obs_buffer: updated obs buffer (or None)
      stu_ctx:        (B, units) — combined spectral context
    """
    new_buffer = jnp.concatenate(
        [buffer[:, 1:, :], action[:, None, :].astype(buffer.dtype)], axis=1)
    units = self.stu_units or self.hidden
    stu_kw = dict(
        num_eigh=self.stu_num_eigh,
        max_seq_len=self.stu_max_seq,
        use_hankel_L=self.stu_use_hankel_L,
        random=self.stu_random,
        random_normalized=self.stu_random_normalized,
        use_neg_bank=self.stu_neg_bank,
        outscale=self.stu_outscale,
        use_shortcut=self.stu_shortcut)
    stu_ctx = self.sub(
        'stu_act', stu.STUCore, units, **stu_kw, **self.kw)(new_buffer)

    new_obs_buffer = None
    if obs_buffer is not None:
      stoch_flat = stoch.reshape((stoch.shape[0], -1))
      new_obs_buffer = jnp.concatenate(
          [obs_buffer[:, 1:, :],
           stoch_flat[:, None, :].astype(obs_buffer.dtype)], axis=1)
      obs_ctx = self.sub(
          'stu_obs', stu.STUCore, units, **stu_kw, **self.kw)(new_obs_buffer)
      stu_ctx = stu_ctx + obs_ctx

    return new_buffer, new_obs_buffer, nn.cast(stu_ctx)

  def _apply_stu(self, deter_seq):
    """Apply STU spectral refinement to a sequence of deter states.
    Returns refined deter with same shape, added as a gated residual."""
    units = self.stu_units or self.hidden
    stu_out = self.sub(
        'stu', stu.STUMixer, units,
        num_eigh=self.stu_num_eigh,
        max_seq_len=self.stu_max_seq,
        use_hankel_L=self.stu_use_hankel_L,
        random=self.stu_random,
        random_normalized=self.stu_random_normalized,
        **self.kw)(nn.cast(deter_seq))
    stu_out = nn.act(self.act)(self.sub('stunorm', nn.Norm, self.norm)(stu_out))
    kw = {**self.kw, 'outscale': 0.01}
    residual = self.sub('stuproj', nn.Linear, self.deter, **kw)(stu_out)
    return nn.cast(deter_seq + residual)

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out


class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    entries = {}
    return carry, entries, tokens


class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons
