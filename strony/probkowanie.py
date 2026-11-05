import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from dsp.sampling import (chwile_probek, czestotliwosc_aliasu, czestotliwosc_dyskretna,
                          czestotliwosci_szczytow, kopie_widma, rekonstrukcja_sinc, sygnal,
                          widmo_dwustronne)
from dsp.signals import FS_ANALOG, FS_AUDIO

ZAPAS_SINC = 0.005
PUNKTY_WYKRESU = 4000

st.title("Próbkowanie")

k1, k2, k3 = st.columns(3)
rodzaj = k1.radio("Sygnał", ["Ton", "Suma tonów"], horizontal=True)
if rodzaj == "Ton":
    tony = [(k2.slider("f [Hz]", 100, 8000, 1000, step=50), 1.0)]
else:
    tony = [(k2.slider("f₁ [Hz]", 100, 8000, 1000, step=50), 1.0),
            (k2.slider("f₂ [Hz]", 100, 8000, 3000, step=50), 0.5)]
fs = k3.slider("fs [Hz]", 500, 20000, 8000, step=100)
dlugosc = ui.okno_czasu(5.0, "okno_2", max_ms=20.0)

f_max = max(f for f, _ in tony)
aliasy = [czestotliwosc_aliasu(f, fs) for f, _ in tony]

if f_max > fs / 2:
    st.warning(f"Aliasing: f_max = {f_max} Hz > fs/2 = {fs / 2:.0f} Hz")

t = np.linspace(0, dlugosc, PUNKTY_WYKRESU)
t_gesty = np.arange(int(dlugosc * FS_ANALOG)) / FS_ANALOG
t_analog, x_analog = ui.przerzedz(t_gesty, sygnal(t_gesty, tony), PUNKTY_WYKRESU)
t_probek = chwile_probek(fs, -ZAPAS_SINC, dlugosc + ZAPAS_SINC)
probki = sygnal(t_probek, tony)
x_rek = rekonstrukcja_sinc(probki, t_probek, fs, t)
widoczne = (t_probek >= 0) & (t_probek <= dlugosc)

fig = ui.nowy_wykres("Sygnał, próbki i rekonstrukcja sinc", "t [ms]", "x(t)", wysokosc=420)
fig.add_trace(go.Scatter(x=t_analog * 1e3, y=x_analog, name="sygnał ciągły", line=dict(color="#7f7f7f")))
ui.lodyzki(fig, t_probek[widoczne] * 1e3, probki[widoczne], "próbki", ui.KOLOR_1)
fig.add_trace(go.Scatter(x=t * 1e3, y=x_rek, name="rekonstrukcja sinc",
                         line=dict(color=ui.KOLOR_2, dash="dash")))
st.plotly_chart(fig)

blad_rek = np.sqrt(np.mean((x_rek - sygnal(t, tony)) ** 2) / np.mean(sygnal(t, tony) ** 2))

probki_1s = sygnal(chwile_probek(fs, 0, 1 - 1 / fs), tony)
alias_teoria = sorted(set(aliasy))
alias_fft = czestotliwosci_szczytow(probki_1s, fs, len(alias_teoria))

k1, k2, k3 = st.columns(3)
k1.metric("fs/2", f"{fs / 2:.0f} Hz")
k2.metric("Błąd rekonstrukcji (RMS)", f"{100 * blad_rek:.2f} %")
k3.metric("Próbek na okres f_max", f"{fs / f_max:.2f}")

st.table({
    "f [Hz]": [f for f, _ in tony],
    "k = round(f/fs)": [round(f / fs) for f, _ in tony],
    "|f − k·fs| [Hz]": aliasy,
    "aliasing": ["tak" if f > fs / 2 else "nie" for f, _ in tony],
})
st.table({"alias - wzór [Hz]": alias_teoria, "alias - FFT [Hz]": [f"{v:.0f}" for v in alias_fft]})

f_widma, amp_widma = widmo_dwustronne(probki_1s, fs)
zakres = max(1.5 * fs, 1.2 * f_max)
fk, ak, kk = kopie_widma(f_widma, amp_widma, fs, -zakres, zakres)

fig = ui.nowy_wykres("Widmo sygnału spróbkowanego", "f [kHz]", "amplituda", wysokosc=380)
ui.lodyzki(fig, fk[kk != 0] / 1e3, ak[kk != 0], "kopie co fs", "#999999")
ui.lodyzki(fig, fk[kk == 0] / 1e3, ak[kk == 0], "pasmo podstawowe", ui.KOLOR_1)
f_oryg = np.array([s * f for f, _ in tony for s in (-1, 1)])
a_oryg = np.array([a / 2 for _, a in tony for _ in (-1, 1)])
fig.add_trace(go.Scatter(x=f_oryg / 1e3, y=a_oryg, mode="markers", name="sygnał ciągły ±f",
                         marker=dict(symbol="circle-open", size=13, color=ui.KOLOR_2, line_width=2)))
for granica in (-fs / 2, fs / 2):
    fig.add_vline(x=granica / 1e3, line_dash="dash", line_color=ui.KOLOR_3)
fig.add_annotation(x=fs / 2e3, y=0.6, text="fs/2", showarrow=False, xshift=18)
fig.update_xaxes(range=[-zakres / 1e3, zakres / 1e3])
st.plotly_chart(fig)

maska = (f_widma >= -fs / 2) & (f_widma < fs / 2) & (amp_widma > 1e-3)
fig = ui.nowy_wykres("Widmo w funkcji ω = 2πf/fs", "ω [rad]", "amplituda", wysokosc=320)
ui.lodyzki(fig, czestotliwosc_dyskretna(f_widma[maska], fs), amp_widma[maska], "|X(ω)|", ui.KOLOR_1)
fig.update_xaxes(range=[-np.pi * 1.05, np.pi * 1.05],
                 tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                 ticktext=["−π", "−π/2", "0", "π/2", "π"])
st.plotly_chart(fig)

t_audio = np.arange(FS_AUDIO) / FS_AUDIO
kol1, kol2 = st.columns(2)
with kol1:
    ui.odtwarzacz(sygnal(t_audio, tony), "Oryginał", fs=FS_AUDIO)
with kol2:
    ui.odtwarzacz(probki_1s, f"Po próbkowaniu z fs = {fs} Hz", fs=fs)
