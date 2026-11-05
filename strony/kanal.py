import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from dsp.am import moduluj_am
from dsp.channel import dodaj_szum
from dsp.demod import demoduluj_am, demoduluj_fm, snr_teoria_am, snr_teoria_fm, snr_teoria_ssb, wzorzec
from dsp.fm import moduluj_fm, pasmo_carsona
from dsp.metrics import snr_wyjsciowy, widmo
from dsp.pcm_errors import FS_WE, ber_bpsk_teoria, kanal_bpsk, lancuch_pcm
from dsp.signals import FS, os_czasu, ton, wczytaj_mowe
from dsp.ssb import odbiornik_ssb, ssb_fazowa

A = 1.0
BRZEG = int(0.01 * FS)
SNR_SIATKA = np.arange(-10, 41, 2.5)
BETY = (1, 2, 5)
F_TONU = 1000
EBN0_SIATKA = np.arange(0, 11, 1.0)
LICZBA_BITOW_BPSK = 400_000
NASTAWY_PCM = {"fs": 8000, "n": 8, "mu_law": True, "filtr_aa": True, "f_aa": 3600.0, "filtr_rek": True}

st.title("Kanał z szumem")

w = ui.wiadomosc()
t = w["t"]
fc = ui.fc()
f_max = w["f_max"]

k1, k2, k3 = st.columns(3)
snr_db = k1.slider("SNR [dB]", -5, 40, 15)
m = k2.slider("AM: m", 0.1, 1.0, 1.0, step=0.05)
beta = k3.slider("FM: β", 0.5, 10.0, 5.0, step=0.5)
df = beta * f_max


@st.cache_data(show_spinner="Liczę...")
def symulacja(klucz, f_max, fc, m, df, snr_db):
    t, x = ui.zbuduj_wiadomosc(*klucz)
    rng = np.random.default_rng(1234)
    odebrane = {
        "AM": dodaj_szum(moduluj_am(x, t, fc, m, A), snr_db, rng)[0],
        "SSB": dodaj_szum(ssb_fazowa(x, t, fc), snr_db, rng)[0],
        "FM": dodaj_szum(moduluj_fm(x, t, fc, df, A), snr_db, rng)[0],
    }
    wyjscia = {
        "AM": demoduluj_am(odebrane["AM"], fc, f_max),
        "SSB": odbiornik_ssb(odebrane["SSB"], t, fc, f_max),
        "FM": demoduluj_fm(odebrane["FM"], t, fc, df, f_max),
    }
    return odebrane, wyjscia, wzorzec(x, f_max)


odebrane, wyjscia, ref = symulacja(ui.klucz_wiadomosci(), f_max, fc, m, df, snr_db)
snr = {nazwa: snr_wyjsciowy(y, ref, BRZEG) for nazwa, y in wyjscia.items()}

B_fm = pasmo_carsona(df, f_max)
if fc - B_fm / 2 < 0 or fc + B_fm / 2 > FS / 2:
    st.error("Pasmo FM poza zakresem 0-96 kHz")

k1, k2, k3, k4 = st.columns(4)
k1.metric("SNR wejściowy", f"{snr_db} dB")
k2.metric("SNR wyjściowy AM", f"{snr['AM']:.1f} dB")
k3.metric("SNR wyjściowy SSB", f"{snr['SSB']:.1f} dB", delta=f"{snr['SSB'] - snr['AM']:+.1f} dB")
k4.metric("SNR wyjściowy FM", f"{snr['FM']:.1f} dB", delta=f"{snr['FM'] - snr['AM']:+.1f} dB")

kolory = {"AM": ui.KOLOR_2, "SSB": ui.KOLOR_3, "FM": ui.KOLOR_1}


def dopasuj_skale(y, ref):
    return y * np.dot(ref, y) / np.dot(y, y)


dlugosc = ui.okno_czasu(ui.domyslne_okno_ms(w) * 2, "okno_kanal")
t0 = 1.0 if w["mowa"] else (0.1 if len(t) / FS > 0.2 else 0.0)
fig = ui.nowy_wykres("Wiadomość oryginalna i odtworzona", "t [ms]", "")
tt, yy = ui.fragment(t, ref, t0, dlugosc)
fig.add_trace(go.Scatter(x=tt * 1e3, y=yy, name="oryginał", line=dict(color="#7f7f7f", width=3)))
for nazwa, y in wyjscia.items():
    tt, yy = ui.fragment(t, dopasuj_skale(y, ref), t0, dlugosc)
    fig.add_trace(go.Scatter(x=tt * 1e3, y=yy, name=nazwa, line=dict(color=kolory[nazwa], width=1.5)))
st.plotly_chart(fig)

kolumny = st.columns(4)
with kolumny[0]:
    ui.odtwarzacz(ref, "Oryginał")
for kol, (nazwa, y) in zip(kolumny[1:], wyjscia.items()):
    with kol:
        ui.odtwarzacz(y[BRZEG:-BRZEG], nazwa)

with st.expander("Widmo sygnału odebranego"):
    zakres = max(B_fm, 4 * f_max)
    fig = ui.nowy_wykres("", "f [kHz]", "amplituda [dB]")
    for nazwa, r in odebrane.items():
        f, amp = widmo(r, FS, okno=True)
        maska = (f >= fc - zakres) & (f <= fc + zakres)
        ff, aa = ui.zmniejsz(f[maska], amp[maska])
        fig.add_trace(go.Scatter(x=ff / 1e3, y=20 * np.log10(np.maximum(aa, 1e-6)), name=f"{nazwa} + szum",
                                 line=dict(color=kolory[nazwa], width=1)))
    st.plotly_chart(fig)


@st.cache_data(show_spinner="Liczę krzywe SNR...")
def krzywe_snr(fc, m):
    t = os_czasu(0.5)
    x = ton(t, F_TONU)
    ref = wzorzec(x, F_TONU)
    rng = np.random.default_rng(7)

    def przebieg(s, odbiornik):
        return [snr_wyjsciowy(odbiornik(dodaj_szum(s, q, rng)[0]), ref, BRZEG) for q in SNR_SIATKA]

    wyniki = {
        "AM": przebieg(moduluj_am(x, t, fc, m, A), lambda r: demoduluj_am(r, fc, F_TONU)),
        "SSB": przebieg(ssb_fazowa(x, t, fc), lambda r: odbiornik_ssb(r, t, fc, F_TONU)),
    }
    for b in BETY:
        wyniki[f"FM β={b}"] = przebieg(moduluj_fm(x, t, fc, b * F_TONU, A),
                                       lambda r, b=b: demoduluj_fm(r, t, fc, b * F_TONU, F_TONU))
    return wyniki


wyniki = krzywe_snr(fc, m)
kolory_krzywych = {"AM": ui.KOLOR_2, "SSB": ui.KOLOR_3, "FM β=1": "#9467bd", "FM β=2": ui.KOLOR_4,
                   "FM β=5": ui.KOLOR_1}
teorie = {"AM": snr_teoria_am(SNR_SIATKA, m, F_TONU), "SSB": snr_teoria_ssb(SNR_SIATKA, F_TONU)}
for b in BETY:
    teorie[f"FM β={b}"] = snr_teoria_fm(SNR_SIATKA, b, F_TONU)

fig = ui.nowy_wykres(f"SNR_wy(SNR_we), ton {F_TONU} Hz, ta sama moc nadawana", "SNR_we [dB]", "SNR_wy [dB]",
                     wysokosc=480)
for nazwa, wartosci in wyniki.items():
    fig.add_trace(go.Scatter(x=SNR_SIATKA, y=wartosci, mode="lines+markers", name=nazwa,
                             line=dict(color=kolory_krzywych[nazwa])))
    fig.add_trace(go.Scatter(x=SNR_SIATKA, y=teorie[nazwa], mode="lines", name=f"teoria {nazwa}",
                             showlegend=False, line=dict(color=kolory_krzywych[nazwa], dash="dash", width=1)))
fig.add_vline(x=snr_db, line_dash="dot")
st.plotly_chart(fig)


def pasmo(nazwa):
    if nazwa == "AM":
        return 2 * F_TONU
    if nazwa == "SSB":
        return F_TONU
    return pasmo_carsona(int(nazwa.split("=")[1]) * F_TONU, F_TONU)


i_akt = int(np.argmin(np.abs(SNR_SIATKA - snr_db)))
kol_akt = f"SNR_wy przy {SNR_SIATKA[i_akt]:.1f} dB"
st.table({
    "system": list(wyniki),
    "B [kHz]": [f"{pasmo(nazwa) / 1e3:g}" for nazwa in wyniki],
    kol_akt: [f"{wartosci[i_akt]:.1f}" for wartosci in wyniki.values()],
    "SNR_wy przy 40 dB": [f"{wartosci[-1]:.1f}" for wartosci in wyniki.values()],
})

st.header("Bity PCM przez kanał AWGN (BPSK)")


@st.cache_data(show_spinner=False)
def ber_zmierzony_bpsk():
    rng = np.random.default_rng(5)
    bity = rng.random(LICZBA_BITOW_BPSK) < 0.5
    return [np.mean(kanal_bpsk(bity, e, rng) != bity) for e in EBN0_SIATKA]


fig = ui.nowy_wykres("BER(Eb/N0)", "Eb/N0 [dB]", "BER", wysokosc=380)
ebn0_gesto = np.linspace(0, 10, 200)
fig.add_trace(go.Scatter(x=ebn0_gesto, y=ber_bpsk_teoria(ebn0_gesto), name="Q(√(2Eb/N0))",
                         line=dict(color="#7f7f7f", dash="dash")))
fig.add_trace(go.Scatter(x=EBN0_SIATKA, y=ber_zmierzony_bpsk(), mode="markers", name="pomiar",
                         marker=dict(size=10, color=ui.KOLOR_1)))
fig.update_yaxes(type="log", exponentformat="power")
st.plotly_chart(fig)

ebn0 = st.slider("Eb/N0 [dB]", 0.0, 12.0, 6.0, step=0.5)


@st.cache_data(show_spinner="Przesyłam PCM...")
def pcm_przez_bpsk(ebn0):
    mowa = wczytaj_mowe(FS_WE, f_odciecia=None)
    return mowa, lancuch_pcm(mowa, **NASTAWY_PCM, kanal="BPSK + AWGN", ebn0_db=ebn0)


mowa, wynik_pcm = pcm_przez_bpsk(ebn0)
k1, k2, k3 = st.columns(3)
k1.metric("BER teoretyczny", f"{ber_bpsk_teoria(ebn0):.2e}")
k2.metric("BER zmierzony", f"{wynik_pcm['ber_zmierzony']:.2e}" if wynik_pcm["bledy_bitowe"] else "0")
k3.metric("SNR mowy po PCM (8 kHz, 8 bit, μ-law)", f"{wynik_pcm['snr_pasmo']:.1f} dB")
kol1, kol2 = st.columns(2)
with kol1:
    ui.odtwarzacz(mowa, "Oryginał", fs=FS_WE)
with kol2:
    ui.odtwarzacz(wynik_pcm["wyjscie"], f"PCM po BPSK, Eb/N0 = {ebn0:g} dB", fs=FS_WE)
