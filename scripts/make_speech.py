"""Generuje data/mowa.wav syntezatorem mowy (offline, pyttsx3 + głos polski)."""
from pathlib import Path

import pyttsx3

ZDANIE = "Dzień dobry, to jest krótka próbka mowy do badania modulacji i kodowania."
PLIK = Path(__file__).resolve().parents[1] / "data" / "mowa.wav"

silnik = pyttsx3.init()
polski = [g for g in silnik.getProperty("voices") if "pl" in str(g.languages).lower() or "polish" in g.name.lower()
          or "paulina" in g.name.lower() or "adam" in g.name.lower()]
if not polski:
    raise SystemExit("Brak polskiego głosu - wgraj własny plik WAV w aplikacji.")
silnik.setProperty("voice", polski[0].id)
silnik.setProperty("rate", 180)
PLIK.parent.mkdir(exist_ok=True)
silnik.save_to_file(ZDANIE, str(PLIK))
silnik.runAndWait()
print(f"Zapisano {PLIK} ({polski[0].name})")
