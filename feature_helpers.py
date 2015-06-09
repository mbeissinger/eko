import numpy as np
from numpy.lib.stride_tricks import as_strided
from audio_utils import AudioSegment
from scipy.io.wavfile import read

EPS = 1e-5


def specgram_real(data, NFFT=256, Fs=2, noverlap=128):
    """
    Computes the spectrogram for a real signal.

    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Params:
    data     : 1D numpy array
    NFFT     : number of elements in the window
    Fs       : sample rate
    noverlap : number of elements to overlap each window

    Returns:
    x        : 2D numpy array, frequency x time
    freq     : 1D array of frequency of each row in x

    Note this is a truncating computation e.g. if NFFT=10,
    noverlap=5 and the signal has 23 elements, then the
    last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(data), "Must not pass in complex numbers"

    stride = NFFT - noverlap
    window = np.hanning(NFFT)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.  Seems somewhat arbitrary thus
    # TODO, awni, check on the validity of this scaling.
    scale = window_norm * Fs

    trunc = (len(data) - NFFT) % stride
    x = data[:len(data) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (NFFT, (len(x) - NFFT) // stride + 1)
    nstrides = (x.strides[0], x.strides[0] * stride)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and nfft/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(Fs) / NFFT * np.arange(x.shape[0])

    return (x, freqs)


def get_feat_dim(step, window, max_freq, **kwargs):
    # TODO sanjeev Figure this out!
    feat_dim_map = {(10, 20, 8000): 81, (10, 20, 16000): 161}
    key = (step, window, max_freq)
    assert key in feat_dim_map
    return feat_dim_map[key]


def compute_volume_normalized_feature(audio_file, feature_info):
    audio = AudioSegment.from_wav(audio_file)
    audio = audio.normalize()
    assert audio.normalized, "Audio segment was not normalized"
    return compute_raw_feature(audio.wav_data, feature_info)


def compute_raw_feature(audio_file, feature_info):
    """
    Compute time x frequency feature matrix *without* any normalization,
    striding, or splicing. We *do* convert to log-space.
    """
    # spec is frequency x time
    spec, _ = compute_specgram(
        audio_file,
        step=feature_info['step'],
        window=feature_info['window'],
        max_freq=feature_info['max_freq'])

    # transpose to time x frequency
    return np.log(spec + EPS).astype(np.float32).T


def compute_specgram(audio_file, step=10, window=20, max_freq=8000, shift=0):
    """
    audio_file : audio wav file
    step : step size in ms between windows
    window : window size in ms
    max_freq : keep frequencies below value
    shift: time in ms to be clipped form the audio data,
           use negative values to clip form the end

    Uses default window for specgram (hamming).
    Return value is Frequency x Time matrix.
    """

    Fs, data = read(audio_file)
    return compute_specgram_raw(Fs, data, step, window, max_freq, shift)


def compute_specgram_raw(Fs, data, step, window, max_freq,
                         shift, spec_fn=specgram_real):
    """
    Same as above when you've already read audio_file
    """

    shift = (Fs/1000) * shift
    if shift < 0:
        data[-shift:] = data[:shift]
    elif shift > 0:
        data[:-shift] = data[shift:]

    # If we have two channels, take just the first
    if len(data.shape) == 2:
        data = data[:, 0].squeeze()

    step_sec = step/1000.
    window_sec = window/1000.

    if step_sec > window_sec:
        raise ValueError("Step size must not be greater than window size")

    noverlap = int((window_sec - step_sec) * Fs)
    NFFT = int(window_sec * Fs)
    Pxx, freqs = spec_fn(data, NFFT=NFFT, Fs=Fs, noverlap=noverlap)[0:2]
    try:
        ind = np.where(freqs == max_freq/2)[0][0] + 1
    except IndexError:
        return None, None
    return Pxx[:ind, :], freqs[:ind]
