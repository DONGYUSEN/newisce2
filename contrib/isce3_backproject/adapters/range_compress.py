"""
FFT-based range compression for SAR raw data.

Implements matched filtering: convolve each range line with the
conjugate-reversed chirp replica in the frequency domain.
"""

import numpy as np


def generate_chirp(chirp_slope, pulse_length, sampling_rate):
    """
    Generate a linear FM chirp replica.

    Parameters
    ----------
    chirp_slope : float
        Chirp rate (Hz/s). Negative for down-chirp.
    pulse_length : float
        Pulse duration (seconds).
    sampling_rate : float
        Range sampling rate (Hz).

    Returns
    -------
    np.ndarray
        Complex64 chirp replica, length = round(pulse_length * sampling_rate).
    """
    n_samples = int(np.round(pulse_length * sampling_rate))
    t = np.arange(n_samples) / sampling_rate - pulse_length / 2.0
    phase = np.pi * chirp_slope * t * t
    return np.exp(1j * phase).astype(np.complex64)


def remove_iq_bias(data, i_bias, q_bias):
    """Subtract DC bias from I and Q channels."""
    return data - np.complex64(i_bias + 1j * q_bias)


def range_compress_pulse(echo, chirp):
    """
    Range-compress a single echo line via frequency-domain matched filtering.

    Parameters
    ----------
    echo : np.ndarray
        Complex64 echo (1-D, length N).
    chirp : np.ndarray
        Complex64 chirp replica.

    Returns
    -------
    np.ndarray
        Complex64 compressed pulse, same length as echo.
    """
    n = len(echo)
    nfft = int(2 ** np.ceil(np.log2(n)))
    echo_fft = np.fft.fft(echo, n=nfft)
    chirp_fft = np.conj(np.fft.fft(chirp, n=nfft))
    compressed = np.fft.ifft(echo_fft * chirp_fft)[:n]
    return compressed.astype(np.complex64)


def range_compress_block(raw_block, chirp):
    """
    Range-compress a 2-D block of echo lines (azimuth × range).

    Uses vectorized FFT along the range axis for all lines simultaneously.

    Parameters
    ----------
    raw_block : np.ndarray
        Complex64 array, shape (n_azimuth, n_range).
    chirp : np.ndarray
        Complex64 chirp replica.

    Returns
    -------
    np.ndarray
        Complex64 range-compressed block, same shape as input.
    """
    n_az, n_rg = raw_block.shape
    nfft = int(2 ** np.ceil(np.log2(n_rg)))
    chirp_fft = np.conj(np.fft.fft(chirp, n=nfft))

    raw_fft = np.fft.fft(raw_block, n=nfft, axis=1)
    compressed = np.fft.ifft(raw_fft * chirp_fft[np.newaxis, :], axis=1)[:, :n_rg]
    return compressed.astype(np.complex64)
