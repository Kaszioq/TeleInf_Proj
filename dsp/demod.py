import numpy as np
from scipy import signal

from dsp.signals import FS, FS_AUDIO, zmien_fs

RZAD_LP = 6
RZAD_BP = 4
ZAPAS_LP = 1.5


def filtr_lp(x, f_odciecia, fs=FS):
    sos = signal.butter(RZAD_LP, f_odciecia, fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def filtr_bp(x, f_dol, f_gora, fs=FS):
    f_dol = max(f_dol, 50.0)
    f_gora = min(f_gora, fs / 2 - 50.0)
    sos = signal.butter(RZAD_BP, [f_dol, f_gora], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def demoduluj_am(r, fc, f_wiad, fs=FS, filtr_wejsciowy=True):
    if filtr_wejsciowy:
        r = filtr_bp(r, fc - 2 * f_wiad, fc + 2 * f_wiad, fs)
    obwiednia = np.abs(signal.hilbert(r))
    return filtr_lp(obwiednia - np.mean(obwiednia), ZAPAS_LP * f_wiad, fs)


def demoduluj_fm(r, t, fc, df, f_wiad, fs=FS, filtr_wejsciowy=True):
    if filtr_wejsciowy:
        r = filtr_bp(r, fc - df - 2 * f_wiad, fc + df + 2 * f_wiad, fs)
    pasmo_podstawowe = signal.hilbert(r) * np.exp(-2j * np.pi * fc * t)
    faza = np.unwrap(np.angle(pasmo_podstawowe))
    # różnica centralna: np.diff przesunęłoby wynik o pół próbki
    f_chwilowa = np.gradient(faza) * fs / (2 * np.pi)
    return filtr_lp(f_chwilowa, ZAPAS_LP * f_wiad, fs) / df


def do_odsluchu(x, fs=FS):
    return zmien_fs(x, fs, FS_AUDIO)


def wzorzec(x, f_wiad, fs=FS):
    return filtr_lp(x, ZAPAS_LP * f_wiad, fs)


def _lp_do_potegi_4(f_wiad, fs=FS):
    """|H(f)|⁴, bo sosfiltfilt filtruje dwukrotnie."""
    sos = signal.butter(RZAD_LP, ZAPAS_LP * f_wiad, fs=fs, output="sos")
    f = np.linspace(0, 10 * f_wiad, 20000)
    _, h = signal.sosfreqz(sos, worN=f, fs=fs)
    return f, np.abs(h) ** 4


def snr_teoria_am(snr_we_db, m, fm, px=0.5, fs=FS):
    f, h4 = _lp_do_potegi_4(fm, fs)
    eta = m**2 * px / (1 + m**2 * px)
    snr_we = 10 ** (np.asarray(snr_we_db) / 10)
    return 10 * np.log10(eta * snr_we * (fs / 2) / np.trapezoid(h4, f) * np.interp(fm, f, h4))


def snr_teoria_fm(snr_we_db, beta, fm, px=0.5, fs=FS):
    f, h4 = _lp_do_potegi_4(fm, fs)
    snr_we = 10 ** (np.asarray(snr_we_db) / 10)
    df = beta * fm
    return 10 * np.log10(snr_we * (fs / 2) * df**2 * px * np.interp(fm, f, h4)
                         / np.trapezoid(f**2 * h4, f))


def snr_teoria_ssb(snr_we_db, fm, fs=FS):
    f, h4 = _lp_do_potegi_4(fm, fs)
    snr_we = 10 ** (np.asarray(snr_we_db) / 10)
    return 10 * np.log10(snr_we * (fs / 2) / np.trapezoid(h4, f) * np.interp(fm, f, h4))
