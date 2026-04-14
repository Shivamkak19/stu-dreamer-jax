import math

import jax.numpy as jnp
import ninjax as nj
import numpy as np

import embodied.jax.nets as nn

f32 = jnp.float32


def nearest_power_of_two(n):
  if n <= 1:
    return 1
  return 1 << int(math.ceil(math.log2(n)))


def get_hankel(seq_len, use_hankel_L=False):
  """Build the Hankel matrix Z whose eigenvectors are the spectral filters."""
  entries = np.arange(1, seq_len + 1, dtype=np.float64)
  i_plus_j = entries[:, None] + entries[None, :]
  if use_hankel_L:
    sign = np.power(-1.0, i_plus_j - 2.0) + 1.0
    denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
    return sign * (8.0 / denom)
  return 2.0 / (i_plus_j ** 3 - i_plus_j)


def get_spectral_filters(seq_len, num_eigh, use_hankel_L=False,
                         random=False, random_normalized=False, seed=0):
  """Spectral basis functions from Hankel eigendecomposition.

  random=False (default): true Hankel eigenvectors, scaled by sigma^{1/4}.
  random=True, random_normalized=False: iid Gaussian filters (no structure).
  random=True, random_normalized=True: Haar-random orthonormal columns with
    the SAME column-energy spectrum (sigma^{1/4}) as the Hankel basis.
    Isolates whether the specific eigenvectors matter vs random directions
    with matched energy profile.
  """
  if random:
    rng = np.random.RandomState(seed)
    if random_normalized:
      g = rng.randn(seq_len, num_eigh).astype(np.float64)
      q, r = np.linalg.qr(g)
      q = q * np.sign(np.diag(r))  # de-bias QR -> Haar measure
      hankel = get_hankel(seq_len, use_hankel_L=use_hankel_L)
      sigma, _ = np.linalg.eigh(hankel)
      sigma = sigma[-num_eigh:]
      return (q * np.power(np.maximum(sigma, 1e-8), 0.25)).astype(np.float32)
    return rng.randn(seq_len, num_eigh).astype(np.float32)
  hankel = get_hankel(seq_len, use_hankel_L=use_hankel_L)
  sigma, phi = np.linalg.eigh(hankel)
  sigma, phi = sigma[-num_eigh:], phi[:, -num_eigh:]
  phi = phi * np.power(np.maximum(sigma, 1e-8), 0.25)
  return phi.astype(np.float32)


def fft_convolve(x, filters):
  """Causal FFT convolution along the second-to-last axis."""
  seq_len = x.shape[-2]
  fft_size = nearest_power_of_two(2 * seq_len - 1)
  x_f = jnp.fft.rfft(x.astype(f32), n=fft_size, axis=-2)
  f_f = jnp.fft.rfft(filters.astype(f32), n=fft_size, axis=0)
  y = jnp.fft.irfft(x_f * f_f, n=fft_size, axis=-2)
  return y[..., :seq_len, :]


class STUMixer(nj.Module):
  """Spectral Transform Unit: parameter-free LDS approximation via Hankel
  eigenbasis.  Precomputes spectral filters at init, applies learned
  linear combination through FFT convolution at call time."""

  num_eigh: int = 24
  max_seq_len: int = 64
  use_hankel_L: bool = False
  random: bool = False
  random_normalized: bool = False

  def __init__(self, units, **kw):
    self.units = units
    self.kw = kw
    self._phi = get_spectral_filters(
        self.max_seq_len, self.num_eigh, self.use_hankel_L,
        random=self.random, random_normalized=self.random_normalized)

  def __call__(self, x):
    # x: (..., seq_len, features) in COMPUTE_DTYPE
    T = x.shape[-2]
    phi = jnp.array(self._phi[:T])

    # Project input to STU working dimension
    h = self.sub('in_proj', nn.Linear, self.units, bias=False, **self.kw)(x)

    # Learnable filter coefficients: (num_eigh, units).
    # Zero-init per the STU paper: M matrices start at 0 so the layer is a
    # no-op at step 0 and the spectral component grows only as gradients flow.
    filter_proj = self.value(
        'filter_proj', nn.Initializer('zeros'),
        (self.num_eigh, self.units))

    # Combine spectral basis into per-channel filters: (T, units)
    filters = jnp.einsum('tk,kd->td', phi, filter_proj.astype(f32))

    # FFT convolution over time dimension
    mixed = fft_convolve(h, filters)

    # Output projection
    out = self.sub('out_proj', nn.Linear, self.units, **self.kw)(
        mixed.astype(x.dtype))
    return out


class STUCore(nj.Module):
  """STU operating on a rolling action-history buffer inside the recurrence.

  At each RSSM step, the buffer holds the W most recent action vectors.
  We run causal FFT convolution against the spectral filter bank, take the
  output at the last (= current) timestep, and project it to a context
  vector that feeds into the GRU's `_core` as a 4th input branch.

  Supports:
  - Positive-only or positive+negative spectral banks (use_neg_bank)
  - Configurable output scale (outscale) for controlling init contribution
  - Direct input shortcut from current action (use_shortcut)"""

  num_eigh: int = 24
  max_seq_len: int = 64
  use_hankel_L: bool = False
  random: bool = False
  random_normalized: bool = False
  use_neg_bank: bool = False
  outscale: float = 0.0
  use_shortcut: bool = False

  def __init__(self, units, **kw):
    self.units = units
    self.kw = kw
    self._phi = get_spectral_filters(
        self.max_seq_len, self.num_eigh, self.use_hankel_L,
        random=self.random, random_normalized=self.random_normalized)
    if self.use_neg_bank and not self.random:
      # (-1)^i modulated filters for negative eigenvalues of A (STU eq 4)
      signs = np.power(-1.0, np.arange(self.max_seq_len)).astype(np.float32)
      self._phi_neg = self._phi * signs[:, None]
    else:
      self._phi_neg = None

  def __call__(self, buffer):
    # buffer: (B, W, in_dim) — action history, oldest first, newest last
    W = buffer.shape[-2]
    phi = jnp.array(self._phi[:W])  # (W, K)

    h = self.sub('in_proj', nn.Linear, self.units, bias=False, **self.kw)(buffer)

    # Positive spectral bank
    filter_proj = self.value(
        'filter_proj', nn.Initializer('zeros'),
        (self.num_eigh, self.units))
    filters = jnp.einsum('tk,kd->td', phi, filter_proj.astype(f32))

    # Negative spectral bank (captures oscillatory / negative-eigenvalue dynamics)
    if self._phi_neg is not None:
      phi_neg = jnp.array(self._phi_neg[:W])
      filter_proj_neg = self.value(
          'filter_proj_neg', nn.Initializer('zeros'),
          (self.num_eigh, self.units))
      filters = filters + jnp.einsum('tk,kd->td', phi_neg, filter_proj_neg.astype(f32))

    mixed = fft_convolve(h, filters)  # (B, W, units)
    last = mixed[..., -1, :]  # (B, units)

    # Direct input shortcut: linear path from current action to output
    if self.use_shortcut:
      shortcut = self.sub('shortcut', nn.Linear, self.units, bias=False,
                          **{**self.kw, 'outscale': 0.0})(buffer[..., -1, :])
      last = last + shortcut.astype(last.dtype)

    out_kw = {**self.kw, 'outscale': self.outscale}
    out = self.sub('out_proj', nn.Linear, self.units, **out_kw)(
        last.astype(buffer.dtype))
    return out
