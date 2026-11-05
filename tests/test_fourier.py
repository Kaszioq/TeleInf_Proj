import numpy as np
import pytest

from dsp.fourier import (FALE, fala, moc_harmonicznych, moc_z_definicji, suma_czesciowa,
                         wspolczynniki, wspolczynniki_fft)

PRZYPADKI = [(nazwa, 0.5) for nazwa in FALE] + [("prostokątna", 0.2), ("prostokątna", 0.75)]


@pytest.mark.parametrize("nazwa, wypelnienie", PRZYPADKI)
def test_wspolczynniki_fft_zgodne_z_analitycznymi(nazwa, wypelnienie):
    k = np.arange(51)
    roznica = wspolczynniki_fft(nazwa, 50, wypelnienie) - wspolczynniki(nazwa, k, wypelnienie)
    assert np.max(np.abs(roznica)) < 1e-3


@pytest.mark.parametrize("nazwa, wypelnienie", PRZYPADKI)
def test_parseval(nazwa, wypelnienie):
    assert moc_harmonicznych(nazwa, 20_000, wypelnienie) == pytest.approx(
        moc_z_definicji(nazwa, wypelnienie), abs=1e-3)


def test_znane_moce():
    assert moc_z_definicji("prostokątna") == pytest.approx(1.0)
    assert moc_z_definicji("trójkątna") == pytest.approx(1 / 3, rel=1e-3)
    assert moc_z_definicji("piłokształtna") == pytest.approx(1 / 3, rel=1e-3)
    assert moc_z_definicji("sinus") == pytest.approx(0.5)


def test_prostokat_tylko_nieparzyste_harmoniczne():
    c = wspolczynniki("prostokątna", np.arange(10))
    assert np.allclose(c[2::2], 0)
    assert 2 * abs(c[1]) == pytest.approx(4 / np.pi)


def test_suma_czesciowa_zbiega_poza_skokami():
    f0 = 100
    t = np.linspace(0.1, 0.4, 1000) / f0
    blad = np.max(np.abs(suma_czesciowa("trójkątna", t, f0, 50) - fala("trójkątna", t, f0)))
    assert blad < 0.01


def test_efekt_gibbsa():
    f0 = 100
    t = np.arange(10_000) / 1e6
    przerost = np.max(suma_czesciowa("prostokątna", t, f0, 49)) - 1
    assert przerost == pytest.approx(0.179, abs=0.01)
