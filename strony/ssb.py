import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from dsp.am import moduluj_am
from dsp.demod import demoduluj_am, wzorzec
from dsp.metrics import korelacja, moc, moc_nosnej, moc_w_pasmie, snr_wyjsciowy, widmo
from dsp.signals import F_MAX, FS, os_czasu, ton, wczytaj_mowe
from dsp.ssb import (charakterystyka, demoduluj_koherentnie, dsb_sc, filtr_wstegi, ssb_fazowa,
                     ssb_filtrowa, tlumienie_fazowej_teoria, tlumienie_filtru_teoria,
                     tlumienie_pasmowe, tlumienie_zmierzone)

BRZEG = int(0.01 * FS)
POMIN_STAN_PRZEJSCIOWY = 0.2
FM_POMIAROWE = (100, 200, 300, 500, 1000, 2000, 3000)

st.title("SSB")

w = ui.wiadomosc()
klucz = ui.klucz_wiadomosci()
t, x = w["t"], w["x"]
fc = ui.fc()
f_max = w["f_max"]
wstega = st.radio("Wstęga", ["USB", "LSB"], horizontal=True)
zakres = max(2 * f_max, 3000)


def widmo_db(s):
    f, amp = widmo(s, FS, okno=w["okno"])
    maska = (f >= fc - zakres) & (f <= fc + zakres)
    ff, aa = ui.zmniejsz(f[maska], amp[maska])
    return ff / 1e3, 20 * np.log10(np.maximum(aa, 1e-6))


def tlumienie(s, pomin=0.0):
    if w["czy_ton"]:
        return tlumienie_zmierzone(s, fc, w["fm"], wstega, pomin=pomin)
    return tlumienie_pasmowe(s, fc, F_MAX, wstega, pomin=pomin)


@st.cache_data(show_spinner=False)
def licz_dsb(klucz, fc):
    t, x = ui.zbuduj_wiadomosc(*klucz)
    return dsb_sc(x, t, fc)


@st.cache_data(show_spinner=False)
def licz_fazowa(klucz, fc, wstega, blad_fazy, niezrownowazenie):
    t, x = ui.zbuduj_wiadomosc(*klucz)
    return ssb_fazowa(x, t, fc, wstega, blad_fazy, niezrownowazenie)


@st.cache_data(show_spinner=False)
def licz_filtrowa(klucz, fc, wstega, typ, rzad):
    t, x = ui.zbuduj_wiadomosc(*klucz)
    return ssb_filtrowa(x, t, fc, F_MAX, wstega, typ, rzad)


@st.cache_data(show_spinner=False)
def tlumienie_filtru_od_fm(fc, wstega, typ, rzad):
    t = os_czasu()
    return [tlumienie_zmierzone(ssb_filtrowa(ton(t, f), t, fc, F_MAX, wstega, typ, rzad), fc, f, wstega,
                                pomin=POMIN_STAN_PRZEJSCIOWY) for f in FM_POMIAROWE]


@st.cache_data(show_spinner=False)
def udzial_mowy_ponizej(f_granica):
    mowa = wczytaj_mowe()
    return moc_w_pasmie(mowa, FS, 0, f_granica) / moc(mowa)


s_dsb = licz_dsb(klucz, fc)

st.header("Metoda fazowa")
k1, k2 = st.columns(2)
blad_fazy = k1.slider("Błąd przesuwnika fazowego [°]", 0.0, 20.0, 0.0, step=0.5)
niezrownowazenie = k2.slider("Niezrównoważenie amplitud [%]", 0.0, 20.0, 0.0, step=0.5) / 100
s_fazowa = licz_fazowa(klucz, fc, wstega, blad_fazy, niezrownowazenie)

fig = ui.nowy_wykres("Widmo SSB (metoda fazowa)", "f [kHz]", "amplituda [dB]")
for s, nazwa, kolor in [(s_dsb, "DSB-SC", "#999999"), (s_fazowa, f"SSB {wstega}", ui.KOLOR_1)]:
    ff, aa = widmo_db(s)
    fig.add_trace(go.Scatter(x=ff, y=aa, name=nazwa, line=dict(color=kolor, width=1)))
fig.add_vline(x=fc / 1e3, line_dash="dot")
fig.update_yaxes(range=[-120, 5])
st.plotly_chart(fig)

teoria = tlumienie_fazowej_teoria(blad_fazy, niezrownowazenie)
k1, k2 = st.columns(2)
zmierzone = tlumienie(s_fazowa)
k1.metric("Tłumienie wstęgi niepożądanej (FFT)", "> 150 dB" if zmierzone > 150 else f"{zmierzone:.1f} dB")
k2.metric("|1 + g·e^(jφ)| / |1 − g·e^(jφ)|", "∞" if teoria > 150 else f"{teoria:.1f} dB")

st.header("Metoda filtrowa")
k1, k2 = st.columns(2)
typ = k1.radio("Filtr", ["Butterworth", "FIR"], horizontal=True)
if typ == "Butterworth":
    rzad = k2.slider("Rząd", 2, 20, 8, step=2)
else:
    rzad = k2.slider("Liczba współczynników", 31, 2001, 401, step=10) | 1
filtr = filtr_wstegi(typ, rzad, fc, F_MAX, wstega)
s_filtrowa = licz_filtrowa(klucz, fc, wstega, typ, rzad)

kol1, kol2 = st.columns(2)
with kol1:
    fig = ui.nowy_wykres("DSB-SC i charakterystyka filtru", "f [kHz]", "[dB]")
    ff, aa = widmo_db(s_dsb)
    fig.add_trace(go.Scatter(x=ff, y=aa, name="DSB-SC", line=dict(color="#999999", width=1)))
    f_h = np.linspace(fc - zakres, fc + zakres, 2000)
    h_db = 20 * np.log10(np.maximum(charakterystyka(typ, filtr, f_h), 1e-6))
    fig.add_trace(go.Scatter(x=f_h / 1e3, y=h_db, name="|H(f)|", line=dict(color=ui.KOLOR_2, width=2)))
    fig.update_yaxes(range=[-120, 5])
    st.plotly_chart(fig)
with kol2:
    fig = ui.nowy_wykres("Widmo SSB (metoda filtrowa)", "f [kHz]", "amplituda [dB]")
    ff, aa = widmo_db(s_filtrowa[int(POMIN_STAN_PRZEJSCIOWY * FS):] if w["czy_ton"] else s_filtrowa)
    fig.add_trace(go.Scatter(x=ff, y=aa, name=f"SSB {wstega}", line=dict(color=ui.KOLOR_1, width=1)))
    fig.update_yaxes(range=[-120, 5])
    st.plotly_chart(fig)

fm_os = np.linspace(20, F_MAX, 400)
fig = ui.nowy_wykres("Tłumienie wstęgi niepożądanej w funkcji f_m", "f_m [Hz]", "tłumienie [dB]", wysokosc=380)
fig.add_trace(go.Scatter(x=fm_os, y=[tlumienie_filtru_teoria(typ, filtr, fc, f, wstega) for f in fm_os],
                         name="z charakterystyki", line=dict(color=ui.KOLOR_1)))
fig.add_trace(go.Scatter(x=FM_POMIAROWE, y=tlumienie_filtru_od_fm(fc, wstega, typ, rzad), mode="markers",
                         name="FFT", marker=dict(size=10, color=ui.KOLOR_2)))
fig.add_vrect(x0=0, x1=300, fillcolor=ui.KOLOR_4, opacity=0.12, line_width=0,
              annotation_text="< 300 Hz", annotation_position="top left")
st.plotly_chart(fig)

k1, k2 = st.columns(2)
k1.metric("Tłumienie dla bieżącej wiadomości", f"{tlumienie(s_filtrowa, POMIN_STAN_PRZEJSCIOWY):.1f} dB")
k2.metric("Energia mowy poniżej 300 Hz", f"{100 * udzial_mowy_ponizej(300):.1f} %")

st.header("AM, DSB-SC, SSB")
s_am = moduluj_am(x, t, fc, 1.0)
sygnaly = {"AM (m = 1)": s_am, "DSB-SC": s_dsb, f"SSB {wstega}": licz_fazowa(klucz, fc, wstega, 0.0, 0.0)}
kolumny = st.columns(3)
for kol, (nazwa, s) in zip(kolumny, sygnaly.items()):
    with kol:
        fig = ui.nowy_wykres(nazwa, "f [kHz]", "[dB]", wysokosc=300)
        ff, aa = widmo_db(s)
        fig.add_trace(go.Scatter(x=ff, y=aa, showlegend=False, line=dict(color=ui.KOLOR_1, width=1)))
        fig.update_yaxes(range=[-120, 5])
        st.plotly_chart(fig)

tabela = {"": [], "pasmo [kHz]": [], "P nośnej": [], "P użyteczna": [], "sprawność": []}
for nazwa, s in sygnaly.items():
    p_calk = moc(s)
    p_nosnej = moc_nosnej(s, FS, fc)
    tabela[""].append(nazwa)
    tabela["pasmo [kHz]"].append(f"{(f_max if nazwa.startswith('SSB') else 2 * f_max) / 1e3:g}")
    tabela["P nośnej"].append(f"{p_nosnej:.4f}")
    tabela["P użyteczna"].append(f"{p_calk - p_nosnej:.4f}")
    tabela["sprawność"].append(f"{100 * (p_calk - p_nosnej) / p_calk:.1f} %")
st.table(tabela)

st.header("Demodulacja koherentna")
k1, k2 = st.columns(2)
df = k1.slider("Δf [Hz]", -200, 200, 0, step=5)
theta = k2.slider("θ [°]", 0, 180, 0, step=5)


@st.cache_data(show_spinner="Demoduluję...")
def demodulacje(klucz, fc, wstega, f_max, df, theta):
    t, x = ui.zbuduj_wiadomosc(*klucz)
    s_ssb = ssb_fazowa(x, t, fc, wstega)
    s_dsb = dsb_sc(x, t, fc)
    s_am = moduluj_am(x, t, fc, 1.0)
    return {
        "AM, detektor obwiedni": demoduluj_am(s_am, fc, f_max, filtr_wejsciowy=False),
        "SSB, detektor obwiedni": demoduluj_am(s_ssb, fc, f_max, filtr_wejsciowy=False),
        "SSB, koherentny": demoduluj_koherentnie(s_ssb, t, fc, f_max, df, theta),
        "DSB-SC, koherentny": demoduluj_koherentnie(s_dsb, t, fc, f_max, df, theta),
        "SSB, koherentny bez błędów": demoduluj_koherentnie(s_ssb, t, fc, f_max),
    }


wyjscia = demodulacje(klucz, fc, wstega, f_max, df, theta)
ref = wzorzec(x, f_max)
srodek = slice(BRZEG, -BRZEG)


def wzmocnienie(y):
    return np.dot(y[srodek], ref[srodek]) / np.dot(ref[srodek], ref[srodek])


def format_snr(snr):
    if snr > 100:
        return "> 100"
    return "< −60" if snr < -60 else f"{snr:.1f}"


tabela = {"": [], "P wyj. / P oryg. [dB]": [], "wzmocnienie": [], "SNR wyj. [dB]": [], "korelacja": []}
for nazwa, y in wyjscia.items():
    stosunek_mocy = moc(y[srodek]) / moc(ref[srodek])
    zanikl = stosunek_mocy < 1e-6
    tabela[""].append(nazwa)
    tabela["P wyj. / P oryg. [dB]"].append("< −60" if zanikl else f"{10 * np.log10(stosunek_mocy):.1f}")
    tabela["wzmocnienie"].append(f"{wzmocnienie(y):.3f}")
    tabela["SNR wyj. [dB]"].append("–" if zanikl else format_snr(snr_wyjsciowy(y, ref, BRZEG)))
    tabela["korelacja"].append("–" if zanikl else f"{korelacja(y[srodek], ref[srodek]):.3f}")
st.table(tabela)
if df == 0:
    st.caption(f"cos θ = {np.cos(np.deg2rad(theta)):.3f}")

dlugosc = ui.okno_czasu(ui.domyslne_okno_ms(w) * 2, "okno_ssb")
t0 = 0.5 if len(t) / FS > 1 else 0.1
fig = ui.nowy_wykres("Wiadomość i wyjścia demodulatorów", "t [ms]", "")
for y, nazwa, kolor, szer in [(ref, "oryginał", "#7f7f7f", 3),
                              (wyjscia["SSB, koherentny"], "SSB", ui.KOLOR_1, 1.5),
                              (wyjscia["DSB-SC, koherentny"], "DSB-SC", ui.KOLOR_2, 1.5),
                              (wyjscia["SSB, detektor obwiedni"], "SSB, obwiednia", ui.KOLOR_4, 1)]:
    tt, yy = ui.fragment(t, y, t0, dlugosc)
    fig.add_trace(go.Scatter(x=tt * 1e3, y=yy, name=nazwa, line=dict(color=kolor, width=szer)))
st.plotly_chart(fig)

fig = ui.nowy_wykres("Widmo po demodulacji SSB", "f [kHz]", "amplituda [dB]", wysokosc=320)
for y, nazwa, kolor in [(ref, "oryginał", "#7f7f7f"), (wyjscia["SSB, koherentny"], "SSB", ui.KOLOR_1)]:
    f, amp = widmo(y, FS, okno=True)
    maska = f <= 1.5 * f_max
    ff, aa = ui.zmniejsz(f[maska], amp[maska])
    fig.add_trace(go.Scatter(x=ff / 1e3, y=20 * np.log10(np.maximum(aa, 1e-6)), name=nazwa,
                             line=dict(color=kolor, width=1)))
fig.update_yaxes(range=[-120, 5])
st.plotly_chart(fig)

kolumny = st.columns(4)
odsluchy = [(ref, "Oryginał"), (wyjscia["AM, detektor obwiedni"], "AM"),
            (wyjscia["SSB, koherentny bez błędów"], "SSB poprawnie"),
            (wyjscia["SSB, koherentny"], f"SSB, Δf = {df} Hz, θ = {theta}°")]
for kol, (y, podpis) in zip(kolumny, odsluchy):
    with kol:
        ui.odtwarzacz(y[srodek], podpis)
