import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from dsp.pcm import krok_kwantyzacji, pcm, przeplywnosc, sqnr, sqnr_teoria
from dsp.sampling import chwile_probek, rekonstrukcja_sinc, sygnal
from dsp.signals import FS_ANALOG, FS_AUDIO

ZAPAS_SINC = 0.005
PUNKTY_WYKRESU = 4000
BITY = np.arange(1, 17)

st.title("PCM")

k1, k2, k3 = st.columns(3)
f = k1.slider("f tonu [Hz]", 100, 4000, 997)
amplituda = k1.slider("Amplituda", 0.01, 1.0, 1.0, step=0.01)
fs = k2.select_slider("fs [Hz]", [8000, 16000, 22050, 44100, 48000], value=8000)
n = k2.slider("Liczba bitów n", 1, 16, 4)
kwantyzator = k3.radio("Kwantyzator", ["równomierny", "μ-law"])
mu_law = kwantyzator == "μ-law"
dlugosc = ui.okno_czasu(2.0, "okno_3", max_ms=10.0)

tony = [(f, amplituda)]

t_probek = chwile_probek(fs, 0, 1 - 1 / fs)
x = sygnal(t_probek, tony)
q, kody, x_dekod = pcm(x, n, mu_law)

t0 = 2 * ZAPAS_SINC
t_gesty = t0 + np.arange(int(dlugosc * FS_ANALOG)) / FS_ANALOG
t_analog, x_analog = ui.przerzedz(t_gesty, sygnal(t_gesty, tony), PUNKTY_WYKRESU)
widoczne = (t_probek >= t0) & (t_probek <= t0 + dlugosc)
do_rekonstrukcji = (t_probek >= t0 - ZAPAS_SINC) & (t_probek <= t0 + dlugosc + ZAPAS_SINC)
t_rek = np.linspace(t0, t0 + dlugosc, PUNKTY_WYKRESU)
x_rek = rekonstrukcja_sinc(x_dekod[do_rekonstrukcji], t_probek[do_rekonstrukcji], fs, t_rek)

st.caption("próbkowanie → kwantyzacja → kodowanie → dekodowanie → rekonstrukcja")

fig = ui.nowy_wykres("Sygnał i sygnał skwantowany", "t [ms]", "x", wysokosc=420)
fig.add_trace(go.Scatter(x=t_analog * 1e3, y=x_analog, name="sygnał ciągły", line=dict(color="#7f7f7f")))
fig.add_trace(go.Scatter(x=t_probek[widoczne] * 1e3, y=x[widoczne], mode="markers", name="próbki",
                         marker=dict(color=ui.KOLOR_1, size=6)))
fig.add_trace(go.Scatter(x=t_probek[widoczne] * 1e3, y=x_dekod[widoczne], name="skwantowany",
                         line=dict(color=ui.KOLOR_2, shape="hv")))
fig.add_trace(go.Scatter(x=t_rek * 1e3, y=x_rek, name="rekonstrukcja", line=dict(color=ui.KOLOR_3, dash="dash")))
st.plotly_chart(fig)

fig = ui.nowy_wykres("Błąd kwantyzacji", "t [ms]", "x − x_q", wysokosc=280)
ui.lodyzki(fig, t_probek[widoczne] * 1e3, (x - x_dekod)[widoczne], "błąd", ui.KOLOR_2)
if not mu_law:
    for granica in (-1, 1):
        fig.add_hline(y=granica * krok_kwantyzacji(n) / 2, line_dash="dot")
st.plotly_chart(fig)

sqnr_zmierzony = sqnr(x, x_dekod)
k1, k2, k3 = st.columns(3)
k1.metric("SQNR zmierzony", f"{sqnr_zmierzony:.2f} dB")
k2.metric("6,02·n + 1,76", f"{sqnr_teoria(n):.2f} dB")
k3.metric("R = fs·n", f"{przeplywnosc(fs, n) / 1e3:.1f} kbit/s")


@st.cache_data(show_spinner=False)
def krzywe_sqnr(f, amplituda, fs):
    x = sygnal(chwile_probek(fs, 0, 1 - 1 / fs), [(f, amplituda)])
    return {nazwa: [sqnr(x, pcm(x, b, mu)[2]) for b in BITY]
            for nazwa, mu in [("równomierny", False), ("μ-law", True)]}


fig = ui.nowy_wykres(f"SQNR(n), amplituda {amplituda:.2f}", "n [bit]", "SQNR [dB]", wysokosc=380)
fig.add_trace(go.Scatter(x=BITY, y=sqnr_teoria(BITY), name="6,02·n + 1,76", line=dict(color="#7f7f7f", dash="dash")))
for nazwa, kolor in [("równomierny", ui.KOLOR_1), ("μ-law", ui.KOLOR_4)]:
    fig.add_trace(go.Scatter(x=BITY, y=krzywe_sqnr(f, amplituda, fs)[nazwa], mode="lines+markers",
                             name=nazwa, line=dict(color=kolor)))
fig.add_vline(x=n, line_dash="dot")
st.plotly_chart(fig)

kol1, kol2 = st.columns([3, 2])
with kol1:
    st.table({
        "n": np.arange(10),
        "x[n]": [f"{v:+.4f}" for v in x[:10]],
        "poziom": q[:10],
        "kod": kody[:10],
        "po dekodowaniu": [f"{v:+.4f}" for v in x_dekod[:10]],
    })
with kol2:
    st.table({
        "": ["telefonia", "CD (1 kanał)", "ustawienia"],
        "fs": ["8 kHz", "44,1 kHz", f"{fs / 1e3:g} kHz"],
        "n": [8, 16, n],
        "R": [f"{przeplywnosc(8000, 8) / 1e3:g} kbit/s", f"{przeplywnosc(44100, 16) / 1e3:g} kbit/s",
              f"{przeplywnosc(fs, n) / 1e3:g} kbit/s"],
    })

kol1, kol2 = st.columns(2)
with kol1:
    ui.odtwarzacz(sygnal(np.arange(FS_AUDIO) / FS_AUDIO, tony), "Oryginał", fs=FS_AUDIO)
with kol2:
    ui.odtwarzacz(x_dekod, f"PCM, n = {n}", fs=fs)
