import numpy as np


def sygnal(t, tony):
    """tony: lista par (częstotliwość, amplituda)."""
    return sum(a * np.cos(2 * np.pi * f * t) for f, a in tony)


def chwile_probek(fs, t_od, t_do):
    n = np.arange(int(np.ceil(t_od * fs)), int(np.floor(t_do * fs)) + 1)
    return n / fs


def rekonstrukcja_sinc(probki, t_probek, fs, t):
    return np.sinc(fs * (t[:, None] - t_probek[None, :])) @ probki


def czestotliwosc_aliasu(f, fs):
    return abs(f - round(f / fs) * fs)


def czestotliwosci_szczytow(probki, fs, ile):
    """Częstotliwości `ile` najsilniejszych prążków w [0, fs/2]; probki muszą trwać 1 s."""
    widmo = np.abs(np.fft.rfft(probki))
    f = np.fft.rfftfreq(len(probki), 1 / fs)
    return np.sort(f[np.argsort(widmo)[::-1][:ile]])


def widmo_dwustronne(probki, fs):
    widmo = np.fft.fftshift(np.fft.fft(probki)) / len(probki)
    f = np.fft.fftshift(np.fft.fftfreq(len(probki), 1 / fs))
    return f, np.abs(widmo)


def kopie_widma(f, amp, fs, f_od, f_do, prog=1e-3):
    """Powiela prążki pasma podstawowego co fs; zwraca (f, amp, numer kopii k)."""
    maska = amp > prog
    f, amp = f[maska], amp[maska]
    k_min = int(np.floor(f_od / fs)) - 1
    k_max = int(np.ceil(f_do / fs)) + 1
    wynik = [(f + k * fs, amp, np.full(len(f), k)) for k in range(k_min, k_max + 1)]
    fk, ak, kk = (np.concatenate(cz) for cz in zip(*wynik))
    w_zakresie = (fk >= f_od) & (fk <= f_do)
    return fk[w_zakresie], ak[w_zakresie], kk[w_zakresie]


def czestotliwosc_dyskretna(f, fs):
    """ω = 2πf/fs sprowadzone do [-π, π)."""
    return np.mod(2 * np.pi * np.asarray(f) / fs + np.pi, 2 * np.pi) - np.pi
