import plotly.graph_objects as go
import streamlit as st

import ui
from dsp.metrics import widmo
from dsp.signals import FS

st.title("Sygnał modulujący")

w = ui.wiadomosc()
t, x = w["t"], w["x"]
dlugosc = ui.okno_czasu(ui.domyslne_okno_ms(w), "okno_4")

kol1, kol2 = st.columns(2)
with kol1:
    tt, xx = ui.fragment(t, x, 0.0, dlugosc)
    fig = ui.nowy_wykres("Przebieg", "t [ms]", "x(t)")
    fig.add_trace(go.Scatter(x=tt * 1e3, y=xx, name="x(t)", line=dict(color=ui.KOLOR_1)))
    st.plotly_chart(fig)

with kol2:
    db = st.toggle("dB", key="db_4")
    f, amp = widmo(x, FS, okno=w["okno"])
    st.plotly_chart(ui.wykres_widma(f, amp, 0, 6000, "Widmo", db))

k1, k2, k3 = st.columns(3)
k1.metric("Czas", f"{len(x) / FS:.2f} s")
k2.metric("f_max", f"{w['f_max']} Hz")
k3.metric("P_x", f"{w['px']:.3f}")

ui.odtwarzacz(x)
