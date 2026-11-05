import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import jv

import ui
from dsp.fm import moduluj_fm, pasmo_carsona, prazki_bessela, rodzaj_fm, suma_bessel_kwadrat
from dsp.metrics import moc, moc_w_pasmie, widmo
from dsp.signals import FS

A = 1.0

st.title("Modulacja FM")

w = ui.wiadomosc()
t, x = w["t"], w["x"]
fc = ui.fc()
f_max = w["f_max"]

sposob = st.radio("Parametr", ["β", "Δf"], horizontal=True)
if sposob == "β":
    beta = st.slider("β", 0.1, 10.0, 2.0, step=0.1)
    df = beta * f_max
else:
    df = float(st.slider("Δf [Hz]", 100, 20000, 2000, step=100))
    beta = df / f_max

B = pasmo_carsona(df, f_max)

k1, k2, k3, k4 = st.columns(4)
k1.metric("β", f"{beta:.2f}")
k2.metric("Δf", f"{df / 1e3:.2f} kHz")
k3.metric("B (Carson)", f"{B / 1e3:.2f} kHz")
k4.metric("FM", rodzaj_fm(beta))

if fc - B / 2 < 0 or fc + B / 2 > FS / 2:
    st.error(f"Pasmo {(fc - B / 2) / 1e3:.1f}-{(fc + B / 2) / 1e3:.1f} kHz poza zakresem 0-96 kHz")


@st.cache_data(show_spinner=False)
def licz_fm(klucz, fc, df):
    t, x = ui.zbuduj_wiadomosc(*klucz)
    return moduluj_fm(x, t, fc, df, A)


s = licz_fm(ui.klucz_wiadomosci(), fc, df)

db = st.toggle("dB", key="db_6")
f, amp = widmo(s, FS, okno=w["okno"])
zakres = max(0.75 * B, 4 * f_max)
fig = ui.wykres_widma(f, amp, fc - zakres, fc + zakres, "Widmo", db)
if w["czy_ton"]:
    _, fn, an = prazki_bessela(beta, fc, f_max, A)
    fig.add_trace(go.Scatter(x=fn / 1e3, y=20 * np.log10(np.maximum(an, 1e-6)) if db else an,
                             mode="markers", name="A·|Jₙ(β)|",
                             marker=dict(symbol="circle-open", size=11, color=ui.KOLOR_2, line_width=2)))
fig.add_vrect(x0=(fc - B / 2) / 1e3, x1=(fc + B / 2) / 1e3, fillcolor=ui.KOLOR_3, opacity=0.12,
              line_width=0, annotation_text="Carson", annotation_position="top left")
st.plotly_chart(fig)

p_calk = moc(s)
k1, k2, k3 = st.columns(3)
k1.metric("Moc w paśmie Carsona", f"{100 * moc_w_pasmie(s, FS, fc - B / 2, fc + B / 2) / p_calk:.2f} %")
k2.metric("Moc całkowita", f"{p_calk:.4f}")
if w["czy_ton"]:
    k3.metric("Σ Jₙ²(β)", f"{suma_bessel_kwadrat(beta):.6f}")

    with st.expander("Prążki: FFT vs Bessel"):
        n, fn, an = prazki_bessela(beta, fc, f_max, A, n_max=int(np.ceil(beta)) + 3)
        st.table({
            "n": n,
            "f [kHz]": [f"{v / 1e3:.2f}" for v in fn],
            "A·|Jₙ(β)|": [f"{v:.4f}" for v in an],
            "FFT": [f"{amp[np.argmin(np.abs(f - fi))]:.4f}" for fi in fn],
        })

kol1, kol2 = st.columns(2)
with kol1:
    bb = np.linspace(0, 12, 400)
    fig = ui.nowy_wykres("Jₙ(β)", "β", "Jₙ(β)", wysokosc=380)
    for k in range(6):
        fig.add_trace(go.Scatter(x=bb, y=jv(k, bb), name=f"J{k}", line=dict(width=1.5)))
    fig.add_vline(x=beta, line_color="#7f7f7f", annotation_text=f"β = {beta:.2f}")
    fig.add_vline(x=2.405, line_dash="dot", line_color=ui.KOLOR_2,
                  annotation_text="J₀ = 0", annotation_position="bottom right")
    fig.add_hline(y=0, line_width=0.5)
    st.plotly_chart(fig)

with kol2:
    dlugosc = ui.okno_czasu(ui.domyslne_okno_ms(w), "okno_6")
    tt, ss = ui.fragment(t, s, 0.0, dlugosc, gesciej=8)
    tx, xx = ui.fragment(t, x, 0.0, dlugosc)
    fig = ui.nowy_wykres("Przebieg", "t [ms]", "s(t)", wysokosc=380)
    fig.add_trace(go.Scatter(x=tt * 1e3, y=ss, name="s(t)", line=dict(color=ui.KOLOR_1, width=1)))
    fig.add_trace(go.Scatter(x=tx * 1e3, y=(fc + df * xx) / 1e3, name="f chwilowa [kHz]",
                             yaxis="y2", line=dict(color=ui.KOLOR_2)))
    fig.update_layout(yaxis2=dict(title="f [kHz]", overlaying="y", side="right"))
    st.plotly_chart(fig)
