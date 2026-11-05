import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from dsp import fourier
from dsp.signals import FS_ANALOG, FS_AUDIO

K_PARSEVAL = 20_000

st.title("Szereg Fouriera")

k1, k2, k3 = st.columns(3)
nazwa = k1.selectbox("Fala", fourier.FALE)
f0 = k2.slider("f₀ [Hz]", 100, 2000, 500, step=50)
n = k3.slider("Liczba harmonicznych N", 1, 50, 5)
wypelnienie = 0.5
if nazwa == "prostokątna":
    wypelnienie = st.slider("Wypełnienie [%]", 5, 95, 50) / 100

t = np.arange(int(3 / f0 * FS_ANALOG)) / FS_ANALOG
x = fourier.fala(nazwa, t, f0, wypelnienie)
x_n = fourier.suma_czesciowa(nazwa, t, f0, n, wypelnienie)

fig = ui.nowy_wykres(f"Suma {n} harmonicznych", "t [ms]", "x(t)", wysokosc=400)
for y, etykieta, kolor in [(x, "oryginał", "#7f7f7f"), (x_n, f"N = {n}", ui.KOLOR_2)]:
    tt, yy = ui.przerzedz(t, y)
    fig.add_trace(go.Scatter(x=tt * 1e3, y=yy, name=etykieta, line=dict(color=kolor)))
st.plotly_chart(fig)

k_max = max(n, 10)
k = np.arange(k_max + 1)
a_wzor = fourier.amplitudy(fourier.wspolczynniki(nazwa, k, wypelnienie))
a_fft = fourier.amplitudy(fourier.wspolczynniki_fft(nazwa, k_max, wypelnienie))

fig = ui.nowy_wykres("Amplitudy harmonicznych", "k (f = k·f₀)", "amplituda")
fig.add_trace(go.Bar(x=k, y=a_wzor, name="wzór", marker_color=ui.KOLOR_1, width=0.5))
fig.add_trace(go.Scatter(x=k, y=a_fft, mode="markers", name="FFT",
                         marker=dict(symbol="circle-open", size=11, color=ui.KOLOR_2, line_width=2)))
fig.add_vrect(x0=-0.5, x1=n + 0.5, fillcolor=ui.KOLOR_3, opacity=0.08, line_width=0)
st.plotly_chart(fig)

p_definicja = fourier.moc_z_definicji(nazwa, wypelnienie)
p_parseval = fourier.moc_harmonicznych(nazwa, K_PARSEVAL, wypelnienie)
p_n = fourier.moc_harmonicznych(nazwa, n, wypelnienie)
skok = x.max() - x.min()

k1, k2, k3, k4 = st.columns(4)
k1.metric("P z definicji", f"{p_definicja:.4f}")
k2.metric(f"P = Σ|c_k|² (k ≤ {K_PARSEVAL})", f"{p_parseval:.4f}")
k3.metric(f"Moc w DC + {n} harm.", f"{100 * p_n / p_definicja:.2f} %")
k4.metric("Przerost", f"{100 * max(0.0, x_n.max() - x.max()) / skok:.1f} %")

t_audio = np.arange(FS_AUDIO) / FS_AUDIO
n_audio = min(n, int((FS_AUDIO / 2 - 1) // f0))
ui.odtwarzacz(fourier.suma_czesciowa(nazwa, t_audio, f0, n_audio, wypelnienie), fs=FS_AUDIO)
