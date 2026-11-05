import streamlit as st

from dsp.signals import F_MAX

st.set_page_config(page_title="Symulator AM/FM", layout="wide")

st.sidebar.header("Wiadomość")
typ = st.sidebar.radio("Rodzaj", ["Ton", "Suma tonów", "Mowa", "Chirp", "Plik WAV"])

fm = 1000
tony = ()
wav = None

if typ == "Ton":
    fm = st.sidebar.slider("f_m [Hz]", 100, F_MAX, 1000, step=50)
elif typ == "Suma tonów":
    ile = st.sidebar.radio("Liczba tonów", [2, 3], horizontal=True)
    tony = (
        st.sidebar.slider("f₁ [Hz]", 100, F_MAX, 500, step=50),
        st.sidebar.slider("f₂ [Hz]", 100, F_MAX, 1200, step=50),
    )
    if ile == 3:
        tony += (st.sidebar.slider("f₃ [Hz]", 100, F_MAX, 2500, step=50),)
elif typ == "Plik WAV":
    plik = st.sidebar.file_uploader("Plik WAV (max 5 s)", type=["wav"])
    if plik is not None:
        wav = plik.getvalue()
    else:
        st.sidebar.info("Brak pliku - ton 1 kHz.")

fc_khz = st.sidebar.slider("f_c [kHz]", 10, 60, 40)
st.sidebar.caption("fs = 192 kHz")

st.session_state["wiadomosc"] = (typ, fm, tony, wav)
st.session_state["fc"] = fc_khz * 1000

strony = {
    "Projekt": [
        st.Page("strony/ssb.py", title="1. SSB", default=True),
        st.Page("strony/pcm_mowy.py", title="2. PCM mowy: błędy i nastawy"),
        st.Page("strony/kanal.py", title="3. Kanał z szumem"),
    ],
    "Podstawy": [
        st.Page("strony/fourier.py", title="Szereg Fouriera"),
        st.Page("strony/probkowanie.py", title="Próbkowanie"),
        st.Page("strony/pcm_podstawy.py", title="PCM - podstawy"),
        st.Page("strony/sygnal.py", title="Sygnał modulujący"),
        st.Page("strony/am.py", title="AM"),
        st.Page("strony/fm.py", title="FM"),
    ],
}
st.navigation(strony).run()
