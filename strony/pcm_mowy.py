import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from dsp.metrics import widmo
from dsp.pcm_errors import (CZESTOTLIWOSCI_PROBKOWANIA, FS_WE, KANALY, ber_bpsk_teoria, lancuch_pcm,
                            snr_od_ber, snr_od_bitu, snr_od_fs, snr_od_n, zapisz_csv)
from dsp.sampling import sygnal
from dsp.signals import os_czasu, wczytaj_mowe

BERY = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
WZMOCNIENIA = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
KROKI = [0.5, 0.25, 0.125, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

DOMYSLNE = {
    "pcm_zrodlo": "Mowa", "pcm_f": 1000,
    "pcm_aa": True, "pcm_aa_auto": True, "pcm_f_aa": 3400,
    "pcm_fs": 8000,
    "pcm_tryb": "n bitów", "pcm_n": 8, "pcm_krok": 0.01, "pcm_wzm": 1.0, "pcm_kwant": "μ-law",
    "pcm_kod": "NKB",
    "pcm_kanal": "bez błędów", "pcm_ber": 1e-3, "pcm_bit": 1, "pcm_ebn0": 6.0,
    "pcm_rek": True,
}
SCENARIUSZE = {
    "Dobre nastawy": {},
    "Za niskie fs, bez AA": {"pcm_fs": 4000, "pcm_aa": False},
    "Bez filtru rekonstrukcyjnego": {"pcm_rek": False},
    "Duży krok (3 bity)": {"pcm_n": 3, "pcm_kwant": "równomierna"},
    "Przesterowanie": {"pcm_wzm": 8.0},
    "BER 1e-3": {"pcm_kanal": "losowe błędy (BER)", "pcm_ber": 1e-3},
    "Tylko MSB": {"pcm_kanal": "tylko bit k", "pcm_bit": 1, "pcm_ber": 1e-2},
    "Tylko LSB": {"pcm_kanal": "tylko bit k", "pcm_bit": 8, "pcm_ber": 1e-2},
    "Cicha mowa, równomierna": {"pcm_wzm": 0.03, "pcm_kwant": "równomierna"},
    "Cicha mowa, μ-law": {"pcm_wzm": 0.03},
}

for klucz, wartosc in DOMYSLNE.items():
    st.session_state.setdefault(klucz, wartosc)


def ustaw(zmiany):
    st.session_state.update({**DOMYSLNE, **zmiany})


def ustaw_fs(fs):
    st.session_state["pcm_fs"] = fs


st.title("PCM mowy: błędy i nastawy")

kolumny = st.columns(5)
for i, (nazwa, zmiany) in enumerate(SCENARIUSZE.items()):
    kolumny[i % 5].button(nazwa, on_click=ustaw, args=(zmiany,), use_container_width=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown("**Źródło**")
    zrodlo = st.radio("Sygnał", ["Mowa", "Ton", "Suma tonów"], key="pcm_zrodlo")
    if zrodlo != "Mowa":
        st.slider("f [Hz]", 100, 8000, step=50, key="pcm_f")
    st.markdown("**Filtr antyaliasingowy**")
    st.toggle("Filtr AA", key="pcm_aa")
    st.toggle("Odcięcie = 0,45·fs", key="pcm_aa_auto", disabled=not st.session_state["pcm_aa"])
    st.slider("Odcięcie AA [Hz]", 300, 20000, step=100, key="pcm_f_aa",
              disabled=not st.session_state["pcm_aa"] or st.session_state["pcm_aa_auto"])
with k2:
    st.markdown("**Próbkowanie**")
    st.select_slider("fs [Hz]", CZESTOTLIWOSCI_PROBKOWANIA, key="pcm_fs")
    p1, p2, p3 = st.columns(3)
    for kol, fs_preset in zip((p1, p2, p3), (4000, 8000, 16000)):
        kol.button(f"{fs_preset // 1000} kHz", on_click=ustaw_fs, args=(fs_preset,), use_container_width=True)
    st.markdown("**Kwantyzacja**")
    tryb = st.radio("Ustaw", ["n bitów", "krok Δ"], horizontal=True, key="pcm_tryb")
    if tryb == "n bitów":
        st.slider("n [bit]", 1, 16, key="pcm_n")
    else:
        st.select_slider("Δ", KROKI, key="pcm_krok")
with k3:
    st.select_slider("Wzmocnienie wejściowe [×]", WZMOCNIENIA, key="pcm_wzm")
    st.radio("Charakterystyka", ["równomierna", "μ-law"], horizontal=True, key="pcm_kwant")
    st.markdown("**Kodowanie**")
    st.radio("Kod", ["NKB", "Gray"], horizontal=True, key="pcm_kod")
    st.markdown("**Rekonstrukcja**")
    st.toggle("Filtr rekonstrukcyjny (off = ZOH)", key="pcm_rek")
with k4:
    st.markdown("**Kanał bitowy**")
    kanal = st.radio("Kanał", KANALY, key="pcm_kanal")
    if kanal in ("losowe błędy (BER)", "tylko bit k"):
        st.select_slider("BER", BERY, format_func=lambda b: f"{b:.0e}", key="pcm_ber")
    if kanal == "tylko bit k":
        st.slider("Bit k (1 = MSB)", 1, 16, key="pcm_bit")
    if kanal == "BPSK + AWGN":
        st.slider("Eb/N0 [dB]", 0.0, 12.0, step=0.5, key="pcm_ebn0")

stan = st.session_state
nastawy = {
    "fs": stan["pcm_fs"],
    "n": stan["pcm_n"],
    "krok": stan["pcm_krok"] if stan["pcm_tryb"] == "krok Δ" else None,
    "filtr_aa": stan["pcm_aa"],
    "f_aa": 0.45 * stan["pcm_fs"] if stan["pcm_aa_auto"] else float(stan["pcm_f_aa"]),
    "wzmocnienie": stan["pcm_wzm"],
    "mu_law": stan["pcm_kwant"] == "μ-law",
    "kod": stan["pcm_kod"],
    "kanal": stan["pcm_kanal"],
    "ber": stan["pcm_ber"],
    "bit_k": stan["pcm_bit"],
    "ebn0_db": stan["pcm_ebn0"],
    "filtr_rek": stan["pcm_rek"],
}


@st.cache_data(show_spinner=False)
def zrodlo_sygnalu(rodzaj, f):
    if rodzaj == "Mowa":
        return wczytaj_mowe(FS_WE, f_odciecia=None)
    t = os_czasu(1.0, FS_WE)
    tony = [(f, 0.9)] if rodzaj == "Ton" else [(f, 0.6), (min(3 * f, 20000), 0.3)]
    return sygnal(t, tony)


@st.cache_data(show_spinner="Liczę łańcuch PCM...")
def symuluj(rodzaj, f, nastawy):
    return lancuch_pcm(zrodlo_sygnalu(rodzaj, f), **nastawy)


x = zrodlo_sygnalu(zrodlo, stan["pcm_f"])
wynik = symuluj(zrodlo, stan["pcm_f"], nastawy)
y = wynik["wyjscie"]
fs = nastawy["fs"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("SNR (pełne pasmo)", f"{wynik['snr']:.1f} dB")
k2.metric("SNR w paśmie 0-fs/2", f"{wynik['snr_pasmo']:.1f} dB")
k3.metric("SQNR", f"{wynik['sqnr']:.1f} dB")
ber_tekst = f"{wynik['ber_zmierzony']:.2e}" if wynik["bledy_bitowe"] else "0"
k4.metric("BER zmierzony", ber_tekst, help=f"błędnych bitów: {wynik['bledy_bitowe']}")
k5.metric("R = fs·n", f"{wynik['przeplywnosc'] / 1e3:g} kbit/s", help=f"n = {wynik['n']} bit, "
          f"{wynik['liczba_poziomow']} poziomów")
if nastawy["kanal"] == "BPSK + AWGN":
    st.caption(f"BER teoretyczny Q(√(2Eb/N0)) = {ber_bpsk_teoria(nastawy['ebn0_db']):.2e}")

t = np.arange(len(x)) / FS_WE
dlugosc = ui.okno_czasu(20.0, "okno_pcm_mowy", max_ms=100.0)
t0 = 1.0 if zrodlo == "Mowa" else 0.1
maska = (t >= t0) & (t <= t0 + dlugosc)
t_probek = np.arange(len(wynik["probki_odebrane"])) / fs
maska_probek = (t_probek >= t0) & (t_probek <= t0 + dlugosc)

fig = ui.nowy_wykres("Oryginał i sygnał odtworzony", "t [ms]", "x", wysokosc=400)
fig.add_trace(go.Scatter(x=t[maska] * 1e3, y=x[maska], name="oryginał", line=dict(color="#7f7f7f", width=3)))
fig.add_trace(go.Scatter(x=t[maska] * 1e3, y=y[maska], name="odtworzony", line=dict(color=ui.KOLOR_2)))
fig.add_trace(go.Scatter(x=t_probek[maska_probek] * 1e3, y=wynik["probki_odebrane"][maska_probek],
                         mode="markers", name="próbki po dekodowaniu", marker=dict(color=ui.KOLOR_1, size=5)))
st.plotly_chart(fig)

kol1, kol2 = st.columns(2)
with kol1:
    f_widma, amp_x = widmo(x, FS_WE, okno=True)
    _, amp_y = widmo(y, FS_WE, okno=True)
    f_gorna = min(FS_WE / 2, max(3 * fs, 12000))
    fig = ui.wykres_widma(f_widma, amp_x, 0, f_gorna, "Widmo", db=True, nazwa="oryginał", kolor="#7f7f7f")
    widoczne = f_widma <= f_gorna
    ff, aa = ui.zmniejsz(f_widma[widoczne], amp_y[widoczne])
    fig.add_trace(go.Scatter(x=ff / 1e3, y=20 * np.log10(np.maximum(aa, 1e-6)), name="odtworzony",
                             line=dict(color=ui.KOLOR_2, width=1)))
    for k in range(1, int(f_gorna // (fs / 2)) + 1):
        fig.add_vline(x=k * fs / 2e3, line_dash="dash" if k == 1 else "dot", line_color=ui.KOLOR_3,
                      annotation_text="fs/2" if k == 1 else None)
    st.plotly_chart(fig)
with kol2:
    fig = ui.nowy_wykres("Błąd", "t [ms]", "odtworzony − oryginał")
    fig.add_trace(go.Scatter(x=t[maska] * 1e3, y=(y - x[: len(y)])[maska[: len(y)]], name="błąd",
                             line=dict(color=ui.KOLOR_2, width=1)))
    st.plotly_chart(fig)

kol1, kol2 = st.columns(2)
with kol1:
    ui.odtwarzacz(x, "Oryginał", fs=FS_WE)
with kol2:
    ui.odtwarzacz(y, "Odtworzony", fs=FS_WE)

st.header("Wykresy zbiorcze")


@st.cache_data(show_spinner="Liczę wykresy zbiorcze...")
def wykresy_zbiorcze(rodzaj, f, nastawy):
    x = zrodlo_sygnalu(rodzaj, f)
    bez_bledow = {**nastawy, "kanal": "bez błędów"}
    return {
        "snr_od_n": snr_od_n(x, bez_bledow),
        "snr_od_ber": snr_od_ber(x, nastawy),
        "snr_od_fs": snr_od_fs(x, bez_bledow),
        "snr_od_bitu": snr_od_bitu(x, nastawy),
    }


zbiorcze = wykresy_zbiorcze(zrodlo, stan["pcm_f"], nastawy)
if st.button("Zapisz do results/*.csv"):
    sciezki = [zapisz_csv(f"{nazwa}.csv", dane) for nazwa, dane in zbiorcze.items()]
    st.success(", ".join(p.name for p in sciezki))

kol1, kol2 = st.columns(2)
with kol1:
    dane = zbiorcze["snr_od_n"]
    fig = ui.nowy_wykres("SQNR(n)", "n [bit]", "SQNR [dB]")
    for nazwa, kolor, styl in [("6,02n + 1,76", "#7f7f7f", "dash"), ("równomierna", ui.KOLOR_1, None),
                               ("μ-law", ui.KOLOR_4, None)]:
        fig.add_trace(go.Scatter(x=dane["n"], y=dane[nazwa], name=nazwa, mode="lines+markers",
                                 line=dict(color=kolor, dash=styl)))
    st.plotly_chart(fig)

    dane = zbiorcze["snr_od_fs"]
    fig = ui.nowy_wykres("SNR(fs) w paśmie 0-fs/2", "fs [kHz]", "SNR [dB]")
    for nazwa, kolor in [("z filtrem AA", ui.KOLOR_3), ("bez filtru AA", ui.KOLOR_2)]:
        fig.add_trace(go.Scatter(x=np.array(dane["fs"]) / 1e3, y=dane[nazwa], name=nazwa, mode="lines+markers",
                                 line=dict(color=kolor)))
    fig.update_xaxes(type="log")
    st.plotly_chart(fig)
with kol2:
    dane = zbiorcze["snr_od_ber"]
    fig = ui.nowy_wykres("SNR(BER)", "BER", "SNR [dB]")
    for nazwa, kolor in [("NKB", ui.KOLOR_1), ("Gray", ui.KOLOR_4)]:
        fig.add_trace(go.Scatter(x=dane["BER"], y=dane[nazwa], name=nazwa, mode="lines+markers",
                                 line=dict(color=kolor)))
    fig.update_xaxes(type="log", exponentformat="power")
    st.plotly_chart(fig)

    dane = zbiorcze["snr_od_bitu"]
    fig = ui.nowy_wykres("SNR przy przekłamaniu bitu k (BER 1e-2)", "k (1 = MSB)", "SNR [dB]")
    fig.add_trace(go.Bar(x=dane["k"], y=dane["SNR"], marker_color=ui.KOLOR_2, name="SNR"))
    st.plotly_chart(fig)
