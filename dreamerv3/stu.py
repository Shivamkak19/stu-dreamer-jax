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


def get_spectral_filters(seq_len, num_eigh, use_hankel_L=False):
  """Hankel eigendecomposition for spectral basis functions (numpy)."""
  entries = np.arange(1, seq_len + 1, dtype=np.float64)
  i_plus_j = entries[:, None] + entries[None, :]
  if use_hankel_L:
    sign = np.power(-1.0, i_plus_j - 2.0) + 1.0
    denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
    hankel = sign * (8.0 / denom)
  else:
    hankel = 2.0 / (i_plus_j ** 3 - i_plus_j)
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

  def __init__(self, units, **kw):
    self.units = units
    self.kw = kw
    self._phi = get_spectral_filters(
        self.max_seq_len, self.num_eigh, self.use_hankel_L)

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

  This is the Run B integration: at each RSSM step, the buffer holds the
  W most recent (masked, dict-concat) action vectors. We run a causal FFT
  convolution against the spectral filter bank, take the convolution output
  at the last (= current) timestep, and project it down to a context vector
  that gets added as a fourth input branch into the GRU's `_core`.

  The output projection is zero-initialized so the spectral context is
  exactly zero at step 0 — i.e. the model is bit-identical to the baseline
  RSSM at initialization, and STU only contributes as gradients flow."""

  num_eigh: int = 24
  max_seq_len: int = 64
  use_hankel_L: bool = False

  def __init__(self, units, **kw):
    self.units = units
    self.kw = kw
    self._phi = get_spectral_filters(
        self.max_seq_len, self.num_eigh, self.use_hankel_L)

  def __call__(self, buffer):
    # buffer: (B, W, in_dim) — action history, oldest at index 0, newest at -1
    W = buffer.shape[-2]
    phi = jnp.array(self._phi[:W])  # (W, K)

    h = self.sub('in_proj', nn.Linear, self.units, bias=False, **self.kw)(buffer)

    filter_proj = self.value(
        'filter_proj', nn.Initializer('zeros'),
        (self.num_eigh, self.units))
    filters = jnp.einsum('tk,kd->td', phi, filter_proj.astype(f32))  # (W, units)

    mixed = fft_convolve(h, filters)  # (B, W, units)
    last = mixed[..., -1, :]  # (B, units) — spectral context at the current step

    out_kw = {**self.kw, 'outscale': 0.0}
    out = self.sub('out_proj', nn.Linear, self.units, **out_kw)(
        last.astype(buffer.dtype))
    return out
