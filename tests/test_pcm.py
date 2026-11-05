import numpy as np
import pytest

from dsp.pcm import (dekoduj, ekspansja_mu, koduj, kompresja_mu, kwantyzuj, pcm, poziom,
                     przeplywnosc, sqnr, sqnr_teoria)

FS = 8000
F_TONU = 997


def ton_pelnej_skali(amplituda=1.0):
    t = np.arange(FS) / FS
    return amplituda * np.sin(2 * np.pi * F_TONU * t)


@pytest.mark.parametrize("n", range(4, 13))
def test_sqnr_zgodny_ze_wzorem(n):
    x = ton_pelnej_skali()
    _, _, x_kwant = pcm(x, n)
    # dla n = 4 odchyłka od wzoru wynosi 0,53 dB
    assert sqnr(x, x_kwant) == pytest.approx(sqnr_teoria(n), abs=0.6)


def test_blad_kwantyzacji_nie_wiekszy_niz_pol_kroku():
    x = np.linspace(-1, 1, 10_001)
    n = 6
    assert np.max(np.abs(x - poziom(kwantyzuj(x, n), n))) <= 2 / 2**n / 2 + 1e-12


def test_kodowanie_i_dekodowanie():
    q = np.array([0, 1, 5, 15])
    kody = koduj(q, 4)
    assert kody == ["0000", "0001", "0101", "1111"]
    assert np.array_equal(dekoduj(kody), q)


def test_mu_law_odwracalne():
    x = np.linspace(-1, 1, 101)
    assert np.allclose(ekspansja_mu(kompresja_mu(x)), x)


def test_mu_law_lepsze_dla_cichego_sygnalu():
    x = ton_pelnej_skali(0.01)
    _, _, rowny = pcm(x, 8)
    _, _, mu = pcm(x, 8, mu_law=True)
    assert sqnr(x, mu) > sqnr(x, rowny) + 10


def test_przeplywnosc():
    assert przeplywnosc(8000, 8) == 64_000
    assert przeplywnosc(44_100, 16) == 705_600
