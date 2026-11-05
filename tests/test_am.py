import pytest

from dsp.am import moce_am_teoria, moduluj_am, sprawnosc_ton
from dsp.metrics import moc, moc_nosnej, moc_prazka
from dsp.signals import FS, os_czasu, ton

FC = 40_000
FM = 1_000


@pytest.mark.parametrize("m", [0.2, 0.5, 0.8, 1.0])
def test_moce_am_zgodne_ze_wzorem(m):
    t = os_czasu()
    s = moduluj_am(ton(t, FM), t, FC, m, A=2.0)
    p_nosnej_wzor, p_wstegi_wzor, eta_wzor = moce_am_teoria(m, A=2.0)

    p_nosnej = moc_prazka(s, FS, FC)
    p_wstegi = moc_prazka(s, FS, FC - FM) + moc_prazka(s, FS, FC + FM)

    assert p_nosnej == pytest.approx(p_nosnej_wzor, rel=0.01)
    assert p_wstegi == pytest.approx(p_wstegi_wzor, rel=0.01)
    assert p_wstegi / moc(s) == pytest.approx(eta_wzor, rel=0.01)


def test_sprawnosc_ton():
    assert sprawnosc_ton(1.0) == pytest.approx(1 / 3)
    assert sprawnosc_ton(0.0) == 0.0


def test_tylko_trzy_prazki():
    t = os_czasu()
    s = moduluj_am(ton(t, FM), t, FC, 0.7)
    p_reszty = moc(s) - sum(moc_prazka(s, FS, f) for f in (FC - FM, FC, FC + FM))
    assert p_reszty < 1e-9


def test_moc_nosnej_przy_niecalkowitej_liczbie_okresow():
    t = os_czasu(0.73)
    s = moduluj_am(ton(t, 333.3), t, 40_003.7, 0.5)
    assert moc_nosnej(s, FS, 40_003.7) == pytest.approx(0.5, rel=1e-3)
