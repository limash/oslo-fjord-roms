import numpy as np

def stretching(Vstretching, theta_s, theta_b, hc, N, kgrid, report=False):
    if Vstretching not in [1, 2, 3, 4, 5]:
        raise ValueError(f'Illegal parameter Vstretching = {Vstretching}')

    s = np.zeros(N)
    C = np.zeros(N)
    ds = 1.0 / N

    if kgrid == 1:
        Nlev = N + 1
        lev = np.arange(Nlev)
    else:
        Nlev = N
        lev = np.arange(1, N + 1) - 0.5

    s = (lev - N) * ds

    if Vstretching == 1:
        if theta_s > 0:
            Ptheta = np.sinh(theta_s * s) / np.sinh(theta_s)
            Rtheta = np.tanh(theta_s * (s + 0.5)) / (2 * np.tanh(0.5 * theta_s)) - 0.5
            C = (1 - theta_b) * Ptheta + theta_b * Rtheta
        else:
            C = s

    elif Vstretching == 2:
        alfa = 1.0
        beta = 1.0
        if theta_s > 0:
            Csur = (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
            if theta_b > 0:
                Cbot = -1 + np.sinh(theta_b * (s + 1)) / np.sinh(theta_b)
                weight = (s + 1)**alfa * (1 + (alfa / beta) * (1 - (s + 1)**beta))
                C = weight * Csur + (1 - weight) * Cbot
            else:
                C = Csur
        else:
            C = s

    return s, C


def set_depth(Vtransform, Vstretching, theta_s, theta_b, hc, N, igrid, h, zeta=None, report=1):
    if Vtransform not in [1, 2]:
        raise ValueError(f'Illegal parameter Vtransform = {Vtransform}')

    if Vstretching not in [1, 2, 3, 4, 5]:
        raise ValueError(f'Illegal parameter Vstretching = {Vstretching}')

    if hc > np.min(h) and Vtransform == 1:
        raise ValueError(f'Critical depth exceeds minimum bathymetry value. Vtransform = {Vtransform}, hc = {hc}, hmax = {np.min(h)}')

    if zeta is None:
        zeta = np.zeros_like(h)

    Np = N + 1
    Lp, Mp = h.shape
    L, M = Lp - 1, Mp - 1

    # Compute vertical stretching function, C(k)
    s, C = stretching(Vstretching, theta_s, theta_b, hc, N, igrid == 5, report)

    # Average bathymetry and free-surface at requested C-grid type
    if igrid == 1 or igrid == 5:
        hr = h
        zetar = zeta
    elif igrid == 2:
        hp = 0.25 * (h[:-1, :-1] + h[1:, :-1] + h[:-1, 1:] + h[1:, 1:])
        zetap = 0.25 * (zeta[:-1, :-1] + zeta[1:, :-1] + zeta[:-1, 1:] + zeta[1:, 1:])
    elif igrid == 3:
        hu = 0.5 * (h[:-1, :] + h[1:, :])
        zetau = 0.5 * (zeta[:-1, :] + zeta[1:, :])
    elif igrid == 4:
        hv = 0.5 * (h[:, :-1] + h[:, 1:])
        zetav = 0.5 * (zeta[:, :-1] + zeta[:, 1:])

    z = np.zeros((Lp, Mp, N))

    # Compute depths
    for k in range(N):
        if Vtransform == 1:
            if igrid in [1, 5]:
                z0 = (s[k] - C[k]) * hc + C[k] * hr
                z[:, :, k] = z0 + zetar * (1.0 + z0 / hr)
            elif igrid == 2:
                z0 = (s[k] - C[k]) * hc + C[k] * hp
                z[:, :, k] = z0 + zetap * (1.0 + z0 / hp)
            # ... include cases for igrid 3 and 4
        elif Vtransform == 2:
            if igrid in [1, 5]:
                z0 = (hc * s[k] + C[k] * hr) / (hc + hr)
                z[:, :, k] = zetar + (zetar + hr) * z0
            elif igrid == 2:
                z0 = (hc * s[k] + C[k] * hp) / (hc + hp)
                z[:, :, k] = zetap + (zetap + hp) * z0
            elif igrid == 3:
                z0 = (hc * s[k] + C[k] * hu) / (hc + hu)
                z[:, :, k] = zetau + (zetau + hu) * z0
            elif igrid == 4:
                z0 = (hc * s[k] + C[k] * hv) / (hc + hv)
                z[:, :, k] = zetav + (zetav + hv) * z0

    return z
