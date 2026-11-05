import numpy as np
import pytest

from dsp.sampling import (chwile_probek, czestotliwosc_aliasu, czestotliwosc_dyskretna,
                          czestotliwosci_szczytow, rekonstrukcja_sinc, sygnal)


@pytest.mark.parametrize("f, fs, oczekiwana", [
    (1000, 8000, 1000),
    (5000, 8000, 3000),
    (7000, 8000, 1000),
    (9000, 8000, 1000),
    (3000, 2000, 1000),
    (6500, 1000, 500),
])
def test_alias_ze_wzoru(f, fs, oczekiwana):
    assert czestotliwosc_aliasu(f, fs) == oczekiwana


@pytest.mark.parametrize("f, fs", [(1000, 8000), (5000, 8000), (7300, 3000), (7900, 2500), (450, 600)])
def test_alias_zmierzony_fft(f, fs):
    t = chwile_probek(fs, 0, 1 - 1 / fs)
    probki = sygnal(t, [(f, 1.0)])
    assert czestotliwosci_szczytow(probki, fs, 1)[0] == pytest.approx(czestotliwosc_aliasu(f, fs))


def test_rekonstrukcja_sinc_bez_aliasingu():
    tony = [(1000, 1.0), (2500, 0.5)]
    fs = 8000
    t_probek = chwile_probek(fs, 0, 0.2)
    probki = sygnal(t_probek, tony)
    t = np.linspace(0.09, 0.11, 2000)
    blad = rekonstrukcja_sinc(probki, t_probek, fs, t) - sygnal(t, tony)
    assert np.sqrt(np.mean(blad**2)) / np.sqrt(np.mean(sygnal(t, tony) ** 2)) < 0.01


def test_rekonstrukcja_z_aliasingiem_daje_alias():
    fs = 8000
    t_probek = chwile_probek(fs, 0, 0.2)
    probki = sygnal(t_probek, [(7000, 1.0)])
    t = np.linspace(0.09, 0.11, 2000)
    odtworzony = rekonstrukcja_sinc(probki, t_probek, fs, t)
    assert np.max(np.abs(odtworzony - sygnal(t, [(1000, 1.0)]))) < 0.05


def test_czestotliwosc_dyskretna():
    assert czestotliwosc_dyskretna(2000, 8000) == pytest.approx(np.pi / 2)
    assert czestotliwosc_dyskretna(7000, 8000) == pytest.approx(-np.pi / 4)
