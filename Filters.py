import numpy as np

def W(u): # The tri-cube kernel
    if u >= 0 and u < 1:
        return (1 - u ** 3) ** 3
    else:
        return 0
def TrendFilter(X):
    lam = len(X)
    k_lam = (lam - 1 - np.arange(lam)) / lam  # weight vector
    for i in range(lam):
        k_lam[i] = W(k_lam[i])
    y = np.sum(k_lam * X) / np.sum(k_lam)
    return y

def SymTrendFilter(X, m_p):
    trend = np.zeros(len(X))
    for i in range(len(X)):
        left = max(0, i - m_p + 1)
        right = min(len(X) - 1, i + m_p - 1)
        lam = right - left + 1
        k_lam = np.abs((np.arange(lam) - (i - left))) / lam
        for j in range(lam):
            k_lam[j] = W(k_lam[j])
        trend[i] = np.sum(k_lam * X[left: right + 1]) / np.sum(k_lam)
    return trend


def SeasonalityFilter(D, m, lam):
    # Return the seasonal part of period m
    C_D = np.zeros(len(D))
    for k in range(m):
        c = D[k]
        i = 0
        while True:
            C_D[k + i * m] = c
            i += 1
            if k + i * m >= len(D):
                break
            c = lam * D[k + i * m] + (1 - lam) * c
    return C_D