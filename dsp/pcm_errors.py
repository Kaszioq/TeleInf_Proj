import csv
from pathlib import Path

import numpy as np
from scipy.special import erfc

from dsp.metrics import snr_db, snr_wyjsciowy
from dsp.pcm import (bity_dla_poziomow, ekspansja_mu, gray_na_nkb, kompresja_mu, kwantyzuj_poziomy,
                     na_bity, nkb_na_gray, wartosc_poziomu, z_bitow)
from dsp.signals import dolnoprzepustowy, zmien_fs

FS_WE = 96_000
CZESTOTLIWOSCI_PROBKOWANIA = (2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 32000, 48000)
KANALY = ("bez błędów", "losowe błędy (BER)", "tylko bit k", "BPSK + AWGN")
KATALOG_WYNIKOW = Path(__file__).resolve().parents[1] / "results"


def przeklam_losowo(bity, ber, rng):
    return bity ^ (rng.random(bity.shape) < ber)


def przeklam_bit(bity, k, ber, rng):
    """k = 1 to MSB."""
    wynik = bity.copy()
    wynik[:, k - 1] ^= rng.random(len(bity)) < ber
    return wynik


def kanal_bpsk(bity, ebn0_db, rng):
    symbole = 1.0 - 2.0 * bity
    sigma = np.sqrt(1 / (2 * 10 ** (ebn0_db / 10)))
    return symbole + rng.normal(0.0, sigma, symbole.shape) < 0


def ber_bpsk_teoria(ebn0_db):
    """Q(√(2Eb/N0)) = ½·erfc(√(Eb/N0))"""
    return 0.5 * erfc(np.sqrt(10 ** (np.asarray(ebn0_db) / 10)))


def lancuch_pcm(x, fs, n=8, filtr_aa=True, f_aa=3400.0, krok=None, wzmocnienie=1.0, mu_law=False,
                kod="NKB", kanal="bez błędów", ber=0.0, bit_k=1, ebn0_db=10.0, filtr_rek=True,
                fs_we=FS_WE, ziarno=0):
    """x próbkowany z fs_we (sygnał 'analogowy'); fs_we musi być wielokrotnością fs."""
    rng = np.random.default_rng(ziarno)
    co_ile = fs_we // fs

    x_aa = dolnoprzepustowy(x, f_aa, fs_we) if filtr_aa and f_aa < fs_we / 2 else x
    probki = x_aa[::co_ile]

    y = np.clip(wzmocnienie * probki, -1, 1)
    if mu_law:
        y = kompresja_mu(y)
    liczba_poziomow = 2**n if krok is None else max(2, int(round(2 / krok)))
    n = bity_dla_poziomow(liczba_poziomow)
    q = kwantyzuj_poziomy(y, liczba_poziomow)

    slowa = nkb_na_gray(q) if kod == "Gray" else q
    bity = na_bity(slowa, n)
    if kanal == "losowe błędy (BER)":
        odebrane = przeklam_losowo(bity, ber, rng)
    elif kanal == "tylko bit k":
        odebrane = przeklam_bit(bity, min(bit_k, n), ber, rng)
    elif kanal == "BPSK + AWGN":
        odebrane = kanal_bpsk(bity, ebn0_db, rng)
    else:
        odebrane = bity

    slowa_odebrane = z_bitow(odebrane)
    q_odebrane = gray_na_nkb(slowa_odebrane) if kod == "Gray" else slowa_odebrane
    q_odebrane = np.minimum(q_odebrane, liczba_poziomow - 1)

    def na_sygnal(numery):
        v = wartosc_poziomu(numery, liczba_poziomow)
        return (ekspansja_mu(v) if mu_law else v) / wzmocnienie

    probki_kwant = na_sygnal(q)
    probki_odebrane = na_sygnal(q_odebrane)
    if filtr_rek:
        wyjscie = zmien_fs(probki_odebrane, fs, fs_we)
    else:
        wyjscie = np.repeat(probki_odebrane, co_ile)
    wyjscie = wyjscie[: len(x)]

    narazone = len(bity) if kanal == "tylko bit k" else bity.size
    brzeg = fs_we // 100
    x_w_pasmie = dolnoprzepustowy(x, 0.45 * fs, fs_we)
    return {
        "probki": probki,
        "probki_odebrane": probki_odebrane,
        "wyjscie": wyjscie,
        "n": n,
        "liczba_poziomow": liczba_poziomow,
        "sqnr": snr_db(probki, probki - probki_kwant),
        "snr": snr_wyjsciowy(wyjscie, x[: len(wyjscie)], obetnij=brzeg),
        "snr_pasmo": snr_wyjsciowy(wyjscie, x_w_pasmie[: len(wyjscie)], obetnij=brzeg),
        "bledy_bitowe": int(np.sum(bity != odebrane)),
        "ber_zmierzony": np.sum(bity != odebrane) / narazone,
        "przeplywnosc": fs * n,
    }


def snr_od_n(x, nastawy, bity=range(1, 17)):
    wynik = {"n": list(bity)}
    for nazwa, mu in [("równomierna", False), ("μ-law", True)]:
        wynik[nazwa] = [lancuch_pcm(x, **{**nastawy, "n": n, "krok": None, "mu_law": mu})["sqnr"] for n in bity]
    wynik["6,02n + 1,76"] = [6.02 * n + 1.76 for n in bity]
    return wynik


def snr_od_ber(x, nastawy, bery=np.logspace(-6, -1, 11)):
    wynik = {"BER": list(bery)}
    for kod in ("NKB", "Gray"):
        wynik[kod] = [lancuch_pcm(x, **{**nastawy, "kod": kod, "kanal": "losowe błędy (BER)", "ber": b})["snr_pasmo"]
                      for b in bery]
    return wynik


def snr_od_fs(x, nastawy, czestotliwosci=CZESTOTLIWOSCI_PROBKOWANIA):
    wynik = {"fs": list(czestotliwosci)}
    for nazwa, aa in [("z filtrem AA", True), ("bez filtru AA", False)]:
        wynik[nazwa] = [lancuch_pcm(x, **{**nastawy, "fs": fs, "filtr_aa": aa, "f_aa": 0.45 * fs})["snr_pasmo"]
                        for fs in czestotliwosci]
    return wynik


def snr_od_bitu(x, nastawy, ber=1e-2):
    n = lancuch_pcm(x, **nastawy)["n"]
    return {"k": list(range(1, n + 1)),
            "SNR": [lancuch_pcm(x, **{**nastawy, "kanal": "tylko bit k", "bit_k": k, "ber": ber})["snr_pasmo"]
                    for k in range(1, n + 1)]}


def zapisz_csv(nazwa, kolumny, katalog=KATALOG_WYNIKOW):
    katalog.mkdir(exist_ok=True)
    sciezka = katalog / nazwa
    with open(sciezka, "w", newline="", encoding="utf-8") as plik:
        zapis = csv.writer(plik)
        zapis.writerow(kolumny.keys())
        zapis.writerows(zip(*kolumny.values()))
    return sciezka
