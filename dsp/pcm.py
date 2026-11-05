import numpy as np

from dsp.metrics import snr_db

MU = 255


def krok_kwantyzacji(n):
    return 2 / 2**n


def kwantyzuj_poziomy(x, liczba_poziomow):
    """Numer poziomu 0 … L−1 dla kwantyzatora równomiernego na [-1, 1]."""
    krok = 2 / liczba_poziomow
    return np.clip(np.floor((x + 1) / krok), 0, liczba_poziomow - 1).astype(int)


def wartosc_poziomu(q, liczba_poziomow):
    return -1 + (np.asarray(q) + 0.5) * 2 / liczba_poziomow


def kwantyzuj(x, n):
    return kwantyzuj_poziomy(x, 2**n)


def poziom(q, n):
    return wartosc_poziomu(q, 2**n)


def bity_dla_poziomow(liczba_poziomow):
    return max(1, int(np.ceil(np.log2(liczba_poziomow))))


def kompresja_mu(x, mu=MU):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def ekspansja_mu(y, mu=MU):
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def koduj(q, n):
    return [format(int(v), f"0{n}b") for v in q]


def dekoduj(kody):
    return np.array([int(k, 2) for k in kody])


def nkb_na_gray(q):
    q = np.asarray(q)
    return q ^ (q >> 1)


def gray_na_nkb(g):
    q = np.array(g, copy=True)
    przesuniecie = np.asarray(g) >> 1
    while np.any(przesuniecie):
        q ^= przesuniecie
        przesuniecie >>= 1
    return q


def na_bity(c, n):
    """Macierz (liczba próbek × n), kolumna 0 to MSB."""
    wagi = 1 << np.arange(n - 1, -1, -1)
    return (np.asarray(c)[:, None] & wagi) > 0


def z_bitow(bity):
    n = bity.shape[1]
    return bity.astype(int) @ (1 << np.arange(n - 1, -1, -1))


def pcm(x, n, mu_law=False):
    """Zwraca (numery poziomów, słowa kodowe, sygnał po dekodowaniu)."""
    y = kompresja_mu(x) if mu_law else x
    q = kwantyzuj(y, n)
    kody = koduj(q, n)
    y_dekod = poziom(dekoduj(kody), n)
    return q, kody, ekspansja_mu(y_dekod) if mu_law else y_dekod


def sqnr(x, x_kwant):
    return snr_db(x, x - x_kwant)


def sqnr_teoria(n):
    return 6.02 * np.asarray(n) + 1.76


def przeplywnosc(fs, n):
    return fs * n
