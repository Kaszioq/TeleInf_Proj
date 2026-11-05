import numpy as np
import pytest

from dsp.metrics import korelacja, moc
from dsp.signals import F_MAX, os_czasu, ton
from dsp.ssb import (demoduluj_koherentnie, dsb_sc, filtr_wstegi, ssb_fazowa, ssb_filtrowa,
                     tlumienie_fazowej_teoria, tlumienie_filtru_teoria, tlumienie_zmierzone)

FC = 40_000
FM = 1_000
BRZEG = 2000


@pytest.mark.parametrize("wstega", ["USB", "LSB"])
def test_fazowa_idealna_tlumi_ponad_40_db(wstega):
    t = os_czasu()
    s = ssb_fazowa(ton(t, FM), t, FC, wstega)
    assert tlumienie_zmierzone(s, FC, FM, wstega) > 40


@pytest.mark.parametrize("wstega", ["USB", "LSB"])
@pytest.mark.parametrize("blad_fazy, niezrownowazenie", [(5, 0), (0, 0.1), (10, 0.05), (20, 0.2), (2, 0.01)])
def test_fazowa_z_niedoskonalosciami_zgodna_ze_wzorem(wstega, blad_fazy, niezrownowazenie):
    t = os_czasu()
    s = ssb_fazowa(ton(t, FM), t, FC, wstega, blad_fazy, niezrownowazenie)
    assert tlumienie_zmierzone(s, FC, FM, wstega) == pytest.approx(
        tlumienie_fazowej_teoria(blad_fazy, niezrownowazenie), abs=1.0)


@pytest.mark.parametrize("typ, rzedy", [("Butterworth", [2, 4, 8, 16]), ("FIR", [51, 201, 801, 1601])])
def test_filtrowa_tlumienie_rosnie_z_rzedem(typ, rzedy):
    t = os_czasu()
    x = ton(t, 500)
    zmierzone = [tlumienie_zmierzone(ssb_filtrowa(x, t, FC, F_MAX, "USB", typ, r), FC, 500, pomin=0.2)
                 for r in rzedy]
    assert all(np.diff(zmierzone) > 0)


def test_filtrowa_pomiar_zgodny_z_charakterystyka():
    t = os_czasu()
    filtr = filtr_wstegi("Butterworth", 8, FC, F_MAX)
    s = ssb_filtrowa(ton(t, 800), t, FC, F_MAX, "USB", "Butterworth", 8)
    assert tlumienie_zmierzone(s, FC, 800, pomin=0.2) == pytest.approx(
        tlumienie_filtru_teoria("Butterworth", filtr, FC, 800), abs=1.0)


@pytest.mark.parametrize("modulacja", ["SSB", "DSB-SC"])
def test_demodulacja_koherentna_bez_bledow(modulacja):
    t = os_czasu(0.5)
    x = ton(t, FM)
    s = ssb_fazowa(x, t, FC) if modulacja == "SSB" else dsb_sc(x, t, FC)
    y = demoduluj_koherentnie(s, t, FC, FM)
    assert korelacja(y[BRZEG:-BRZEG], x[BRZEG:-BRZEG]) > 0.99


def test_dsb_sc_przy_90_stopniach_zanika():
    t = os_czasu(0.5)
    x = ton(t, FM)
    y = demoduluj_koherentnie(dsb_sc(x, t, FC), t, FC, FM, theta_deg=90)
    assert moc(y[BRZEG:-BRZEG]) < 1e-4 * moc(x)


def test_dsb_sc_tlumione_o_cos_theta():
    t = os_czasu(0.5)
    x = ton(t, FM)
    y = demoduluj_koherentnie(dsb_sc(x, t, FC), t, FC, FM, theta_deg=60)
    wzmocnienie = np.dot(y[BRZEG:-BRZEG], x[BRZEG:-BRZEG]) / np.dot(x[BRZEG:-BRZEG], x[BRZEG:-BRZEG])
    assert wzmocnienie == pytest.approx(np.cos(np.deg2rad(60)), abs=0.02)
