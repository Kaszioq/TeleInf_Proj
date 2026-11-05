import numpy as np
import pytest

from dsp.fm import moduluj_fm, pasmo_carsona, prazki_bessela, suma_bessel_kwadrat
from dsp.metrics import moc, moc_prazka, moc_w_pasmie
from dsp.signals import FS, os_czasu, ton

FC = 40_000
FM = 1_000
BETY = [0.5, 1.0, 2.0, 2.405, 5.0, 10.0]


def fm_ton(beta):
    t = os_czasu()
    return moduluj_fm(ton(t, FM), t, FC, beta * FM)


@pytest.mark.parametrize("beta", BETY)
def test_prazki_zgodne_z_bessel(beta):
    s = fm_ton(beta)
    _, f, amp_teoria = prazki_bessela(beta, FC, FM)
    amp = np.array([np.sqrt(2 * moc_prazka(s, FS, fi)) for fi in f])
    assert np.max(np.abs(amp - amp_teoria)) < 0.01


@pytest.mark.parametrize("beta", BETY)
def test_suma_bessel_kwadrat(beta):
    assert suma_bessel_kwadrat(beta) == pytest.approx(1.0, abs=1e-9)


def test_moc_fm_niezalezna_od_beta():
    assert np.allclose([moc(fm_ton(b)) for b in BETY], 0.5, rtol=0.01)


@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_moc_w_pasmie_carsona(beta):
    s = fm_ton(beta)
    B = pasmo_carsona(beta * FM, FM)
    assert moc_w_pasmie(s, FS, FC - B / 2, FC + B / 2) / moc(s) >= 0.98
