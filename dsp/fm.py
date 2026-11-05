import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import jv


def moduluj_fm(x, t, fc, df, A=1.0):
    calka_x = cumulative_trapezoid(x, t, initial=0)
    return A * np.cos(2 * np.pi * fc * t + 2 * np.pi * df * calka_x)


def prazki_bessela(beta, fc, fm, A=1.0, n_max=None):
    if n_max is None:
        n_max = int(np.ceil(beta)) + 6
    n = np.arange(-n_max, n_max + 1)
    return n, fc + n * fm, A * np.abs(jv(n, beta))


def suma_bessel_kwadrat(beta):
    n = np.arange(-int(np.ceil(beta)) - 30, int(np.ceil(beta)) + 31)
    return np.sum(jv(n, beta) ** 2)


def pasmo_carsona(df, fm):
    return 2 * (df + fm)


def rodzaj_fm(beta):
    return "wąskopasmowa" if beta < 1 else "szerokopasmowa"
