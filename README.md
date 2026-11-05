# Symulator: SSB, PCM mowy z błędami, kanał z szumem

Interaktywna aplikacja w Pythonie i Streamlit. Projekt zaliczeniowy z telekomunikacji.

## Zakładki

**Projekt**

| zakładka | zawartość |
|---|---|
| 1. SSB | DSB-SC, SSB metodą fazową (błąd fazy, niezrównoważenie, tłumienie wstęgi vs \|1 + g·e^(jφ)\| / \|1 − g·e^(jφ)\|), metodą filtrową (Butterworth / FIR, tłumienie w funkcji f_m), porównanie z AM, demodulacja koherentna z błędem Δf i θ, detektor obwiedni, odsłuch |
| 2. PCM mowy: błędy i nastawy | filtr AA → próbkowanie → kwantyzacja (n / Δ, wzmocnienie, μ-law) → kod NKB / Gray → kanał bitowy (BER, bit k, BPSK) → dekodowanie → rekonstrukcja (filtr / ZOH); scenariusze, wykresy zbiorcze, zapis do `results/*.csv` |
| 3. Kanał z szumem | AM, SSB i FM (β = 1, 2, 5) przy tej samej mocy nadawanej, SNR_wy(SNR_we) z teorią, pasmo, odsłuch; bity PCM przez BPSK + AWGN |

**Podstawy:** szereg Fouriera, próbkowanie, PCM - podstawy, sygnał modulujący, AM, FM.

## Model przeskalowany

Zakładki z nośną: fs = 192 kHz, nośna 10-60 kHz, wiadomość do 4 kHz, odsłuch 48 kHz.
Łańcuch PCM: sygnał „analogowy” 96 kHz, fs = 2-48 kHz (podzielniki 96 kHz).

## Uruchomienie

```bash
pip install -r requirements.txt
streamlit run app.py
```

```bash
pytest
```

Nagranie mowy `data/mowa.wav` pochodzi z syntezatora (offline, głos polski Windows):

```bash
pip install pyttsx3
python scripts/make_speech.py
```

Wyniki wykresów zbiorczych PCM dla dobrych nastaw:

```bash
python scripts/eksperymenty_pcm.py
```

## Struktura

```
dsp/              obliczenia, bez Streamlita
  ssb.py          DSB-SC, SSB fazowa i filtrowa, tłumienie wstęgi, demodulacja koherentna
  pcm_errors.py   łańcuch PCM z błędami, kanał bitowy, BPSK, wykresy zbiorcze, zapis CSV
  pcm.py          kwantyzacja, μ-law, NKB / Gray, bity, SQNR
  signals.py      stałe, generatory, wczytywanie WAV i mowy
  am.py, fm.py, channel.py, demod.py, metrics.py, fourier.py, sampling.py
app.py            pasek boczny i nawigacja (Projekt / Podstawy)
ui.py             wspólne funkcje interfejsu
strony/           po jednym pliku na zakładkę
tests/            testy porównujące symulację z teorią + test dymny wszystkich stron
scripts/          generowanie mowy i wyników
data/mowa.wav     próbka mowy
results/          wyniki eksperymentów PCM (CSV)
```

Katalog stron nazywa się `strony/`, a nie `pages/`, bo Streamlit automatycznie wczytuje `pages/`
i przy wejściu bezpośrednio na podstronę pomija nawigację z `app.py`.
