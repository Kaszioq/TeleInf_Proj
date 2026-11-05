import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import signal
from scipy.io import wavfile

from dsp import signals
from dsp.signals import FS, FS_AUDIO, zmien_fs

KOLOR_1 = "#1f77b4"
KOLOR_2 = "#d62728"
KOLOR_3 = "#2ca02c"
KOLOR_4 = "#ff7f0e"


@st.cache_data(show_spinner=False)
def zbuduj_wiadomosc(typ, fm, tony, wav):
    if typ == "Plik WAV" and wav is not None:
        x = signals.wczytaj_wav(wav)
        return np.arange(len(x)) / FS, x
    if typ == "Mowa":
        x = signals.wczytaj_mowe()
        return np.arange(len(x)) / FS, x

    t = signals.os_czasu()
    if typ == "Suma tonów":
        x = signals.suma_tonow(t, tony, [1.0, 0.7, 0.5][: len(tony)])
    elif typ == "Chirp":
        x = signals.chirp(t)
    else:
        x = signals.ton(t, fm)
    return t, x


def klucz_wiadomosci():
    return st.session_state["wiadomosc"]


def wiadomosc():
    typ, fm, tony, wav = klucz_wiadomosci()
    t, x = zbuduj_wiadomosc(typ, fm, tony, wav)
    czy_ton = typ == "Ton" or (typ == "Plik WAV" and wav is None)
    if czy_ton:
        f_max = fm
    elif typ == "Suma tonów":
        f_max = max(tony)
    else:
        f_max = signals.F_MAX
    return {
        "t": t,
        "x": x,
        "czy_ton": czy_ton,
        "fm": fm,
        "f_max": f_max,
        "mowa": typ == "Mowa" or (typ == "Plik WAV" and wav is not None),
        "okno": not (czy_ton or typ == "Suma tonów"),
        "px": float(np.mean(x**2)),
    }


def fc():
    return st.session_state["fc"]


def zmniejsz(f, y, max_pkt=4000):
    """Maksimum z bloków - prążki widma nie znikają po przerzedzeniu."""
    if len(y) <= max_pkt:
        return f, y
    k = int(np.ceil(len(y) / max_pkt))
    bloki = y[: len(y) // k * k].reshape(-1, k)
    idx = np.argmax(bloki, axis=1) + np.arange(bloki.shape[0]) * k
    return f[idx], y[idx]


def przerzedz(t, y, max_pkt=5000):
    krok = max(1, int(np.ceil(len(y) / max_pkt)))
    return t[::krok], y[::krok]


def fragment(t, y, t0, dlugosc, gesciej=1, fs=FS):
    n0 = int(t0 * fs)
    n1 = min(len(y), n0 + max(int(dlugosc * fs), 2))
    if gesciej == 1:
        return t[n0:n1], y[n0:n1]
    zapas = 32
    a, b = max(0, n0 - zapas), min(len(y), n1 + zapas)
    yy = signal.resample_poly(y[a:b], gesciej, 1)
    tt = t[a] + np.arange(len(yy)) / (fs * gesciej)
    maska = (tt >= t[n0]) & (tt <= t[n1 - 1])
    return tt[maska], yy[maska]


def nowy_wykres(tytul, os_x, os_y, wysokosc=360):
    fig = go.Figure()
    fig.update_layout(
        title=tytul,
        xaxis_title=os_x,
        yaxis_title=os_y,
        height=wysokosc,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


def lodyzki(fig, x, y, nazwa, kolor, symbol="circle", **kwargs):
    """Wykres typu stem: pionowe kreski od zera i znacznik na końcu."""
    x, y = np.asarray(x), np.asarray(y)
    xs = np.column_stack([x, x, np.full_like(x, np.nan, dtype=float)]).ravel()
    ys = np.column_stack([np.zeros_like(y), y, np.full_like(y, np.nan, dtype=float)]).ravel()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=kolor, width=1),
                             showlegend=False, hoverinfo="skip", **kwargs))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=nazwa,
                             marker=dict(color=kolor, size=7, symbol=symbol), **kwargs))


def wykres_widma(f, amp, f_od, f_do, tytul, db=False, nazwa="FFT", kolor=KOLOR_1):
    maska = (f >= f_od) & (f <= f_do)
    ff, aa = zmniejsz(f[maska], amp[maska])
    if db:
        aa = 20 * np.log10(np.maximum(aa, 1e-6))
    fig = nowy_wykres(tytul, "f [kHz]", "amplituda [dB]" if db else "amplituda")
    fig.add_trace(go.Scatter(x=ff / 1e3, y=aa, name=nazwa, line=dict(color=kolor, width=1)))
    if db:
        fig.update_yaxes(range=[-100, 5])
    return fig


def okno_czasu(domyslne_ms, klucz, max_ms=50.0):
    return st.slider("Fragment [ms]", 0.5, max_ms, float(domyslne_ms), step=0.5, key=klucz) / 1e3


def domyslne_okno_ms(w):
    if w["czy_ton"]:
        return min(50.0, max(0.5, round(3000 / w["fm"] * 2) / 2))
    return 10.0


def wav_bajty(x, fs=FS_AUDIO):
    szczyt = np.max(np.abs(x))
    if szczyt > 0:
        x = 0.9 * x / szczyt
    bufor = io.BytesIO()
    wavfile.write(bufor, fs, (x * 32767).astype(np.int16))
    return bufor.getvalue()


def odtwarzacz(x, podpis=None, fs=FS):
    if podpis:
        st.caption(podpis)
    if fs != FS_AUDIO:
        x = zmien_fs(x, fs, FS_AUDIO)
    st.audio(wav_bajty(x), format="audio/wav")
