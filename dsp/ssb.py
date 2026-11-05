import numpy as np
from scipy import signal

from dsp.demod import ZAPAS_LP, filtr_lp
from dsp.metrics import moc_prazka, moc_w_pasmie
from dsp.signals import FS


def dsb_sc(x, t, fc):
    return x * np.cos(2 * np.pi * fc * t)


def ssb_fazowa(x, t, fc, wstega="USB", blad_fazy_deg=0.0, niezrownowazenie=0.0):
    """Gałąź kwadraturowa ma wzmocnienie g = 1 + niezrównoważenie i przesunięcie fazy o błąd."""
    x_hilbert = np.imag(signal.hilbert(x))
    g = 1 + niezrownowazenie
    znak = -1 if wstega == "USB" else 1
    galaz_q = g * x_hilbert * np.sin(2 * np.pi * fc * t + np.deg2rad(blad_fazy_deg))
    return x * np.cos(2 * np.pi * fc * t) + znak * galaz_q


def tlumienie_fazowej_teoria(blad_fazy_deg, niezrownowazenie):
    """Stosunek wstęgi pożądanej do niepożądanej w dB: |1 + g·e^(jφ)| / |1 − g·e^(jφ)|."""
    ge = (1 + niezrownowazenie) * np.exp(1j * np.deg2rad(blad_fazy_deg))
    niepozadana = max(abs(1 - ge), 1e-15)
    return 20 * np.log10(abs(1 + ge) / niepozadana)


def filtr_wstegi(typ, rzad, fc, f_max, wstega="USB", fs=FS):
    pasmo = (fc, fc + f_max) if wstega == "USB" else (fc - f_max, fc)
    if typ == "Butterworth":
        return signal.butter(rzad, pasmo, btype="bandpass", fs=fs, output="sos")
    return signal.firwin(rzad, pasmo, pass_zero=False, fs=fs)


def zastosuj_filtr(typ, filtr, x):
    if typ == "Butterworth":
        return signal.sosfilt(filtr, x)
    return signal.lfilter(filtr, 1.0, x)


def charakterystyka(typ, filtr, f, fs=FS):
    if typ == "Butterworth":
        _, h = signal.sosfreqz(filtr, worN=f, fs=fs)
    else:
        _, h = signal.freqz(filtr, worN=f, fs=fs)
    return np.abs(h)


def ssb_filtrowa(x, t, fc, f_max, wstega="USB", typ="Butterworth", rzad=8):
    filtr = filtr_wstegi(typ, rzad, fc, f_max, wstega)
    return zastosuj_filtr(typ, filtr, dsb_sc(x, t, fc))


def czestotliwosci_wsteg(fc, fm, wstega="USB"):
    """(pożądana, niepożądana)"""
    return (fc + fm, fc - fm) if wstega == "USB" else (fc - fm, fc + fm)


def tlumienie_zmierzone(s, fc, fm, wstega="USB", fs=FS, pomin=0.0):
    s = s[int(pomin * fs):]
    pozadana, niepozadana = czestotliwosci_wsteg(fc, fm, wstega)
    return 10 * np.log10(moc_prazka(s, fs, pozadana) / max(moc_prazka(s, fs, niepozadana), 1e-30))


def tlumienie_pasmowe(s, fc, f_max, wstega="USB", fs=FS, pomin=0.0):
    s = s[int(pomin * fs):]
    gorna = moc_w_pasmie(s, fs, fc, fc + f_max)
    dolna = moc_w_pasmie(s, fs, fc - f_max, fc)
    pozadana, niepozadana = (gorna, dolna) if wstega == "USB" else (dolna, gorna)
    return 10 * np.log10(pozadana / max(niepozadana, 1e-30))


def tlumienie_filtru_teoria(typ, filtr, fc, fm, wstega="USB", fs=FS):
    h = charakterystyka(typ, filtr, np.array(czestotliwosci_wsteg(fc, fm, wstega)), fs)
    return 20 * np.log10(h[0] / max(h[1], 1e-15))


def demoduluj_koherentnie(s, t, fc, f_max, df=0.0, theta_deg=0.0, fs=FS):
    lokalna_nosna = 2 * np.cos(2 * np.pi * (fc + df) * t + np.deg2rad(theta_deg))
    return filtr_lp(s * lokalna_nosna, ZAPAS_LP * f_max, fs)


def filtr_idealny(x, f_od, f_do, fs=FS):
    widmo = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    widmo[(f < f_od) | (f > f_do)] = 0
    return np.fft.irfft(widmo, len(x))


def odbiornik_ssb(r, t, fc, f_max, wstega="USB", fs=FS):
    # filtr idealny: realny filtr z krawędzią w fc tłumiłby najniższe częstotliwości wiadomości
    zapas = 50.0
    pasmo = (fc - zapas, fc + ZAPAS_LP * f_max) if wstega == "USB" else (fc - ZAPAS_LP * f_max, fc + zapas)
    return demoduluj_koherentnie(filtr_idealny(r, *pasmo, fs), t, fc, f_max, fs=fs)
