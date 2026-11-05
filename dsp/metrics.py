import numpy as np


def moc(x):
    return np.mean(np.abs(x) ** 2)


def _jednostronne(widmo, n):
    widmo[1:] *= 2
    if n % 2 == 0:
        widmo[-1] /= 2
    return widmo


def widmo(x, fs, okno=False):
    """Widmo amplitudowe, cos o amplitudzie A daje prążek A."""
    w = np.hanning(len(x)) if okno else np.ones(len(x))
    amp = np.abs(np.fft.rfft(x * w)) / np.sum(w)
    return np.fft.rfftfreq(len(x), 1 / fs), _jednostronne(amp, len(x))


def widmo_mocy(x, fs):
    p = np.abs(np.fft.rfft(x)) ** 2 / len(x) ** 2
    return np.fft.rfftfreq(len(x), 1 / fs), _jednostronne(p, len(x))


def moc_w_pasmie(x, fs, f_od, f_do):
    f, p = widmo_mocy(x, fs)
    tol = 1e-6 * fs / len(x)
    return np.sum(p[(f >= f_od - tol) & (f <= f_do + tol)])


def moc_prazka(x, fs, f0):
    f, p = widmo_mocy(x, fs)
    return p[np.argmin(np.abs(f - f0))]


def moc_nosnej(s, fs, fc):
    """Moc składowej o dokładnie fc (DTFT w jednym punkcie), bez przecieku przy niecałkowitej liczbie okresów."""
    t = np.arange(len(s)) / fs
    amplituda = 2 * np.mean(s * np.exp(-2j * np.pi * fc * t))
    return np.abs(amplituda) ** 2 / 2


def snr_db(sygnal, blad):
    return 10 * np.log10(moc(sygnal) / moc(blad))


def snr_wyjsciowy(y, ref, obetnij=0):
    """Skala y dopasowana do ref metodą najmniejszych kwadratów, reszta to błąd."""
    if obetnij > 0:
        y, ref = y[obetnij:-obetnij], ref[obetnij:-obetnij]
    y = y - np.mean(y)
    ref = ref - np.mean(ref)
    a = np.dot(y, ref) / np.dot(ref, ref)
    return snr_db(a * ref, y - a * ref)


def korelacja(a, b):
    return np.corrcoef(a, b)[0, 1]
