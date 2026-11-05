"""Zapisuje wykresy zbiorcze PCM mowy (dobre nastawy) do results/*.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsp.pcm_errors import FS_WE, snr_od_ber, snr_od_bitu, snr_od_fs, snr_od_n, zapisz_csv  # noqa: E402
from dsp.signals import wczytaj_mowe  # noqa: E402

DOBRE_NASTAWY = {"fs": 8000, "n": 8, "mu_law": True, "filtr_aa": True, "f_aa": 3600.0, "filtr_rek": True}

mowa = wczytaj_mowe(FS_WE, f_odciecia=None)
for nazwa, funkcja in [("snr_od_n", snr_od_n), ("snr_od_ber", snr_od_ber), ("snr_od_fs", snr_od_fs),
                       ("snr_od_bitu", snr_od_bitu)]:
    print(zapisz_csv(f"{nazwa}.csv", funkcja(mowa, DOBRE_NASTAWY)))
