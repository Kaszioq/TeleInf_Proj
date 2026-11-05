import numpy as np


def dodaj_szum(s, snr_db, rng=None):
    rng = rng or np.random.default_rng()
    sigma2 = np.mean(s**2) / 10 ** (snr_db / 10)
    szum = rng.normal(0.0, np.sqrt(sigma2), len(s))
    return s + szum, szum
