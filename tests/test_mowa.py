import numpy as np
import pytest

from dsp.metrics import widmo
from dsp.signals import FS, wczytaj_mowe


@pytest.mark.parametrize("fs", [8000, 48_000, FS])
def test_wczytanie_mowy(fs):
    x = wczytaj_mowe(fs)
    assert 3 <= len(x) / fs <= 5
    assert np.max(np.abs(x)) == pytest.approx(1.0)
    assert abs(np.mean(x)) < 1e-9


def test_mowa_ograniczona_do_4_khz():
    x = wczytaj_mowe(FS)
    f, amp = widmo(x, FS, okno=True)
    assert np.sum(amp[f > 6000] ** 2) < 1e-4 * np.sum(amp**2)
