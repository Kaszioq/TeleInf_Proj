import numpy as np
import pytest

from dsp.metrics import widmo
from dsp.pcm import gray_na_nkb, na_bity, nkb_na_gray, z_bitow
from dsp.pcm_errors import (FS_WE, ber_bpsk_teoria, kanal_bpsk, lancuch_pcm, przeklam_bit,
                            przeklam_losowo, snr_od_bitu)
from dsp.sampling import czestotliwosc_aliasu
from dsp.signals import os_czasu

RNG = np.random.default_rng(0)


def ton(f, czas=1.0):
    t = os_czasu(czas, FS_WE)
    return 0.9 * np.sin(2 * np.pi * f * t)


def test_gray_i_bity_odwracalne():
    q = np.arange(256)
    assert np.array_equal(gray_na_nkb(nkb_na_gray(q)), q)
    assert np.all(np.sum(na_bity(nkb_na_gray(q[1:]), 8) != na_bity(nkb_na_gray(q[:-1]), 8), axis=1) == 1)
    assert np.array_equal(z_bitow(na_bity(q, 8)), q)


@pytest.mark.parametrize("ber", [1e-3, 1e-2, 1e-1])
def test_zmierzony_ber_zgodny_z_zadanym(ber):
    bity = RNG.random((50_000, 8)) < 0.5
    zmierzony = np.mean(przeklam_losowo(bity, ber, RNG) != bity)
    assert zmierzony == pytest.approx(ber, rel=0.1)


def test_przeklamanie_tylko_wybranego_bitu():
    bity = RNG.random((10_000, 8)) < 0.5
    roznice = przeklam_bit(bity, 3, 0.1, RNG) != bity
    assert roznice[:, 2].mean() == pytest.approx(0.1, rel=0.1)
    assert not np.delete(roznice, 2, axis=1).any()


@pytest.mark.parametrize("ebn0_db", [0, 4, 7])
def test_bpsk_zgodny_z_teoria(ebn0_db):
    bity = RNG.random(400_000) < 0.5
    zmierzony = np.mean(kanal_bpsk(bity, ebn0_db, RNG) != bity)
    assert zmierzony == pytest.approx(ber_bpsk_teoria(ebn0_db), rel=0.1)


def test_lancuch_bez_bledow_wierny():
    wynik = lancuch_pcm(ton(1000), fs=8000, n=12)
    assert wynik["snr"] > 40
    assert wynik["ber_zmierzony"] == 0


def test_lancuch_ber_zmierzony():
    wynik = lancuch_pcm(ton(1000), fs=8000, n=8, kanal="losowe błędy (BER)", ber=1e-2)
    assert wynik["ber_zmierzony"] == pytest.approx(1e-2, rel=0.15)


def test_msb_psuje_bardziej_niz_lsb():
    nastawy = {"fs": 8000, "n": 8}
    snr = snr_od_bitu(ton(1000), nastawy, ber=1e-2)["SNR"]
    assert snr[0] < snr[-1] - 20
    assert all(np.diff(snr) > 0)


@pytest.mark.parametrize("f, fs", [(3000, 4000), (5000, 8000), (7000, 12000)])
def test_bez_filtru_aa_pojawia_sie_alias(f, fs):
    alias = czestotliwosc_aliasu(f, fs)

    def amplituda_aliasu(filtr_aa):
        wyjscie = lancuch_pcm(ton(f), fs=fs, n=12, filtr_aa=filtr_aa, f_aa=0.45 * fs)["wyjscie"]
        czest, amp = widmo(wyjscie, FS_WE)
        return amp[np.argmin(np.abs(czest - alias))]

    bez_filtru = amplituda_aliasu(False)
    assert bez_filtru > 0.5
    assert amplituda_aliasu(True) < 0.02 * bez_filtru


def test_przesterowanie_obniza_sqnr():
    x = ton(1000)
    assert lancuch_pcm(x, 8000, n=8, wzmocnienie=4)["sqnr"] < lancuch_pcm(x, 8000, n=8)["sqnr"] - 20


def test_krok_wyznacza_liczbe_bitow():
    wynik = lancuch_pcm(ton(1000), 8000, krok=0.1)
    assert wynik["liczba_poziomow"] == 20
    assert wynik["n"] == 5
