import numpy as np

FALE = ("prostokątna", "trójkątna", "piłokształtna", "sinus")


def fala(nazwa, t, f0, wypelnienie=0.5):
    faza = np.mod(t * f0, 1.0)
    if nazwa == "prostokątna":
        return np.where(faza < wypelnienie, 1.0, -1.0)
    if nazwa == "trójkątna":
        return 1 - 4 * np.abs(faza - np.round(faza))
    if nazwa == "piłokształtna":
        return 2 * faza - 1
    return np.sin(2 * np.pi * faza)


def wspolczynniki(nazwa, k, wypelnienie=0.5):
    """Zespolone współczynniki c_k dla k ≥ 0."""
    k = np.asarray(k)
    c = np.zeros(k.shape, dtype=complex)
    niezerowe = k != 0
    kn = k[niezerowe]

    if nazwa == "prostokątna":
        c[~niezerowe] = 2 * wypelnienie - 1
        c[niezerowe] = (1 - np.exp(-2j * np.pi * kn * wypelnienie)) / (1j * np.pi * kn)
    elif nazwa == "trójkątna":
        c[niezerowe] = np.where(kn % 2 == 1, 4 / (np.pi * kn) ** 2, 0)
    elif nazwa == "piłokształtna":
        c[niezerowe] = 1j / (np.pi * kn)
    else:
        c[k == 1] = -0.5j
    return c


def wspolczynniki_fft(nazwa, k_max, wypelnienie=0.5, probek_na_okres=8192):
    t = np.arange(probek_na_okres) / probek_na_okres
    x = fala(nazwa, t, 1.0, wypelnienie)
    return np.fft.fft(x)[: k_max + 1] / probek_na_okres


def amplitudy(c):
    """Amplitudy jednostronne: |c₀| oraz 2|c_k|."""
    a = 2 * np.abs(c)
    a[0] = np.abs(c[0])
    return a


def suma_czesciowa(nazwa, t, f0, n, wypelnienie=0.5):
    c = wspolczynniki(nazwa, np.arange(n + 1), wypelnienie)
    x = np.full(len(t), c[0].real)
    for k in range(1, n + 1):
        x += 2 * np.real(c[k] * np.exp(2j * np.pi * k * f0 * t))
    return x


def moc_z_definicji(nazwa, wypelnienie=0.5, probek=200_000):
    t = np.arange(probek) / probek
    return np.mean(fala(nazwa, t, 1.0, wypelnienie) ** 2)


def moc_harmonicznych(nazwa, n, wypelnienie=0.5):
    c = wspolczynniki(nazwa, np.arange(n + 1), wypelnienie)
    return np.abs(c[0]) ** 2 + 2 * np.sum(np.abs(c[1:]) ** 2)
