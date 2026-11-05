import numpy as np
import pytest

from dsp.am import moduluj_am
from dsp.channel import dodaj_szum
from dsp.demod import (demoduluj_am, demoduluj_fm, do_odsluchu, snr_teoria_am,
                       snr_teoria_fm, snr_teoria_ssb, wzorzec)
from dsp.fm import moduluj_fm
from dsp.metrics import korelacja, moc, snr_wyjsciowy
from dsp.signals import FS, FS_AUDIO, os_czasu, ton
from dsp.ssb import odbiornik_ssb, ssb_fazowa

FC = 40_000
FM = 1_000
BRZEG = 2000


def test_moc_szumu():
    t = os_czasu()
    s = moduluj_am(ton(t, FM), t, FC, 0.5)
    _, szum = dodaj_szum(s, 10, np.random.default_rng(0))
    assert moc(s) / moc(szum) == pytest.approx(10.0, rel=0.02)


def test_demodulacja_am_bez_szumu():
    t = os_czasu(0.5)
    x = ton(t, FM)
    y = demoduluj_am(moduluj_am(x, t, FC, 0.8), FC, FM)
    assert korelacja(y[BRZEG:-BRZEG], x[BRZEG:-BRZEG]) > 0.99


@pytest.mark.parametrize("beta", [1, 2, 5])
def test_demodulacja_fm_bez_szumu(beta):
    t = os_czasu(0.5)
    x = ton(t, FM)
    y = demoduluj_fm(moduluj_fm(x, t, FC, beta * FM), t, FC, beta * FM, FM)
    assert korelacja(y[BRZEG:-BRZEG], x[BRZEG:-BRZEG]) > 0.99
    assert np.max(np.abs(y[BRZEG:-BRZEG])) == pytest.approx(1.0, rel=0.1)


def test_snr_zgodny_z_teoria_przy_duzym_snr():
    rng = np.random.default_rng(1)
    t = os_czasu(0.5)
    x = ton(t, FM)
    ref = wzorzec(x, FM)
    snr_we = 20

    r, _ = dodaj_szum(moduluj_am(x, t, FC, 1.0), snr_we, rng)
    snr_am = snr_wyjsciowy(demoduluj_am(r, FC, FM), ref, BRZEG)
    assert snr_am == pytest.approx(snr_teoria_am(snr_we, 1.0, FM), abs=1.5)

    r, _ = dodaj_szum(moduluj_fm(x, t, FC, 5 * FM), snr_we, rng)
    snr_fm = snr_wyjsciowy(demoduluj_fm(r, t, FC, 5 * FM, FM), ref, BRZEG)
    assert snr_fm == pytest.approx(snr_teoria_fm(snr_we, 5, FM), abs=1.5)
    assert snr_fm > snr_am + 10


def test_decymacja():
    assert len(do_odsluchu(np.ones(FS))) == FS_AUDIO


def test_ssb_zgodne_z_teoria_i_lepsze_od_am():
    rng = np.random.default_rng(2)
    t = os_czasu(0.5)
    x = ton(t, FM)
    ref = wzorzec(x, FM)
    snr_we = 20

    r, _ = dodaj_szum(ssb_fazowa(x, t, FC), snr_we, rng)
    snr_ssb = snr_wyjsciowy(odbiornik_ssb(r, t, FC, FM), ref, BRZEG)
    assert snr_ssb == pytest.approx(snr_teoria_ssb(snr_we, FM), abs=1.0)

    r, _ = dodaj_szum(moduluj_am(x, t, FC, 1.0), snr_we, rng)
    snr_am = snr_wyjsciowy(demoduluj_am(r, FC, FM), ref, BRZEG)
    assert snr_ssb - snr_am == pytest.approx(10 * np.log10(3), abs=1.0)
