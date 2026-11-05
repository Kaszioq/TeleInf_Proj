import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import hilbert

import ui
from dsp.am import moce_am_teoria, moduluj_am, obwiednia_am, sprawnosc_ton
from dsp.metrics import moc, moc_nosnej, widmo
from dsp.signals import FS

A = 1.0

st.title("Modulacja AM")

w = ui.wiadomosc()
t, x = w["t"], w["x"]
fc = ui.fc()
fm = w["fm"]

m = st.slider("Głębokość modulacji m", 0.0, 1.5, 0.8, step=0.05)


@st.cache_data(show_spinner=False)
def licz_am(klucz, fc, m):
    t, x = ui.zbuduj_wiadomosc(*klucz)
    return moduluj_am(x, t, fc, m, A)


s = licz_am(ui.klucz_wiadomosci(), fc, m)
obwiednia = obwiednia_am(x, m, A)

if m > 1:
    st.warning(f"Przemodulowanie (m = {m:.2f} > 1)")

dlugosc = ui.okno_czasu(ui.domyslne_okno_ms(w), "okno_5")
tt, ss = ui.fragment(t, s, 0.0, dlugosc, gesciej=8)
to, oo = ui.fragment(t, obwiednia, 0.0, dlugosc)

fig = ui.nowy_wykres("Przebieg i obwiednia", "t [ms]", "s(t)")
fig.add_trace(go.Scatter(x=tt * 1e3, y=ss, name="s(t)", line=dict(color=ui.KOLOR_1, width=1)))
fig.add_trace(go.Scatter(x=to * 1e3, y=oo, name="A[1+m·x(t)]", line=dict(color=ui.KOLOR_2, dash="dash")))
fig.add_trace(go.Scatter(x=to * 1e3, y=-oo, showlegend=False, line=dict(color=ui.KOLOR_2, dash="dash")))
if m > 1:
    fig.add_trace(go.Scatter(x=to * 1e3, y=np.where(oo < 0, oo, np.nan), name="obwiednia < 0",
                             line=dict(color="red", width=4)))
st.plotly_chart(fig)

db = st.toggle("dB", key="db_5")
f, amp = widmo(s, FS, okno=w["okno"])
zakres = max(3 * w["f_max"], 3000)
fig = ui.wykres_widma(f, amp, fc - zakres, fc + zakres, "Widmo", db)
if w["czy_ton"]:
    for fp, etykieta in [(fc - fm, "LSB"), (fc, "nośna"), (fc + fm, "USB")]:
        a = amp[np.argmin(np.abs(f - fp))]
        fig.add_annotation(x=fp / 1e3, y=20 * np.log10(max(a, 1e-6)) if db else a,
                           text=etykieta, showarrow=True, arrowhead=2, ay=-35)
else:
    fig.add_vrect(x0=(fc - w["f_max"]) / 1e3, x1=fc / 1e3, fillcolor=ui.KOLOR_3, opacity=0.12,
                  line_width=0, annotation_text="LSB", annotation_position="top left")
    fig.add_vrect(x0=fc / 1e3, x1=(fc + w["f_max"]) / 1e3, fillcolor=ui.KOLOR_4, opacity=0.12,
                  line_width=0, annotation_text="USB", annotation_position="top right")
st.plotly_chart(fig)

p_nosnej_wzor, p_wstegi_wzor, eta_wzor = moce_am_teoria(m, A, px=w["px"])
p_calk = moc(s)
p_nosnej = moc_nosnej(s, FS, fc)
p_wstegi = p_calk - p_nosnej

st.table({
    "": ["P nośnej", "P wstęg", "P całkowita", "η"],
    "wzór": [f"{p_nosnej_wzor:.4f}", f"{p_wstegi_wzor:.4f}", f"{p_nosnej_wzor + p_wstegi_wzor:.4f}",
             f"{100 * eta_wzor:.2f} %"],
    "FFT": [f"{p_nosnej:.4f}", f"{p_wstegi:.4f}", f"{p_calk:.4f}", f"{100 * p_wstegi / p_calk:.2f} %"],
})

kol1, kol2 = st.columns(2)
with kol1:
    mm = np.linspace(0, 1.5, 151)
    fig = ui.nowy_wykres("η(m)", "m", "η [%]", wysokosc=330)
    fig.add_trace(go.Scatter(x=mm, y=100 * sprawnosc_ton(mm), name="m²/(2+m²)", line=dict(color=ui.KOLOR_1)))
    fig.add_trace(go.Scatter(x=[m], y=[100 * sprawnosc_ton(m)], mode="markers", name=f"m = {m:.2f}",
                             marker=dict(size=12, color=ui.KOLOR_2)))
    fig.add_vline(x=1, line_dash="dot")
    st.plotly_chart(fig)

with kol2:
    tt, yy = ui.fragment(t, np.abs(hilbert(s)), 0.0, dlugosc)
    fig = ui.nowy_wykres("Obwiednia z detektora (bez szumu)", "t [ms]", "", wysokosc=330)
    fig.add_trace(go.Scatter(x=to * 1e3, y=oo, name="A[1+m·x(t)]", line=dict(color=ui.KOLOR_3, dash="dash")))
    fig.add_trace(go.Scatter(x=tt * 1e3, y=yy, name="|hilbert(s)|", line=dict(color=ui.KOLOR_2)))
    st.plotly_chart(fig)
