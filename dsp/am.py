import numpy as np


def moduluj_am(x, t, fc, m, A=1.0):
    return A * (1 + m * x) * np.cos(2 * np.pi * fc * t)


def obwiednia_am(x, m, A=1.0):
    return A * (1 + m * x)


def moce_am_teoria(m, A=1.0, px=0.5):
    """Zwraca (moc nośnej, moc wstęg, sprawność); px = średnia z x², dla tonu 1/2."""
    p_nosnej = A**2 / 2
    p_wstegi = A**2 * m**2 * px / 2
    return p_nosnej, p_wstegi, p_wstegi / (p_nosnej + p_wstegi)


def sprawnosc_ton(m):
    m = np.asarray(m, dtype=float)
    return m**2 / (2 + m**2)
