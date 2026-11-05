import io
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

FS = 192_000
FS_AUDIO = 48_000
FS_ANALOG = 1_000_000
F_MAX = 4_000
CZAS = 1.0
MAX_CZAS_WAV = 5.0
PLIK_MOWY = Path(__file__).resolve().parents[1] / "data" / "mowa.wav"


def os_czasu(czas=CZAS, fs=FS):
    return np.arange(int(round(czas * fs))) / fs


def normalizuj(x):
    x = x - np.mean(x)
    szczyt = np.max(np.abs(x))
    return x / szczyt if szczyt > 0 else x


def ton(t, f):
    return np.cos(2 * np.pi * f * t)


def suma_tonow(t, czestotliwosci, amplitudy=None):
    if amplitudy is None:
        amplitudy = [1.0] * len(czestotliwosci)
    return normalizuj(sum(a * ton(t, f) for f, a in zip(czestotliwosci, amplitudy)))


def chirp(t, f0=100, f1=F_MAX):
    return signal.chirp(t, f0=f0, t1=t[-1], f1=f1)


def dolnoprzepustowy(x, f_odciecia, fs=FS, rzad=8):
    sos = signal.butter(rzad, f_odciecia, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def zmien_fs(x, fs_z, fs_na):
    d = gcd(int(fs_z), int(fs_na))
    return signal.resample_poly(x, int(fs_na) // d, int(fs_z) // d)


def wczytaj_wav(dane, fs=FS, max_czas=MAX_CZAS_WAV, f_odciecia=F_MAX):
    """Mono, przycięcie, zmiana fs, opcjonalny filtr LP i normalizacja do max |x| = 1."""
    fs_pliku, x = wavfile.read(io.BytesIO(dane))
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = zmien_fs(x[: int(max_czas * fs_pliku)], fs_pliku, fs)
    if f_odciecia is not None and f_odciecia < fs / 2:
        x = dolnoprzepustowy(x, f_odciecia, fs)
    return normalizuj(x)


def wczytaj_mowe(fs=FS, f_odciecia=F_MAX):
    return wczytaj_wav(PLIK_MOWY.read_bytes(), fs, f_odciecia=f_odciecia)
