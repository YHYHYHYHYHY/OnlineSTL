import numpy as np
import math
from Filters import TrendFilter, SymTrendFilter, SeasonalityFilter


# Ensure that the input length is larger than 4 * m (m represents the largest period)
class OnlineSTL:
    def __init__(self, series, periods, lam=0.7):
        self.periods = periods
        self.m = max(periods)
        self.A = series[-4 * self.m:]  # array to store the latest 4m elements of series
        self.D = series[-4 * self.m:]  # array which represents the completely deseasonalized series of the last 4m elements
        self.lam = lam  # smoothing parameter used for seasonality filtering
        self.K = []  # the seasonal series of period m_p obtained after removing the initial trend of the latest 4m points
        self.epi_S = []  # running smoothed seasonality estimates for m_p of the initially detrended series
        self.epi_T = []  # running smoothed seasonality estimates for m_p of the series obtained after removing initial trend and trend of seasonality

        for m_p in periods:
            trend1 = SymTrendFilter(self.A, m_p)
            T1 = self.A - trend1
            season1 = SeasonalityFilter(T1, m_p, self.lam)
            self.K.append(season1[-4 * self.m:])
            self.epi_S.append(season1[-m_p:])
            trend2 = SymTrendFilter(season1, int(m_p * 3 / 2))
            D5 = T1 - trend2
            season2 = SeasonalityFilter(D5, m_p, self.lam)
            self.epi_T.append(season2[-m_p:])
            Q = season2[-4 * self.m:]
            self.D = self.D - Q

        self.finish_init = True

    def UpdateArray(self, array, x):
        # a simple operation on a circular array that replaces the oldest element in array and with x
        array = np.append(array, x)
        array = np.delete(array, 0)
        return array

    def update(self, x):
        # Input: x -- value of new coming point
        # Output: T -- Trend 
        # S -- Seasonal
        # R -- Remainder 
        self.A = self.UpdateArray(self.A, x)
        b = x
        S = []
        for i in range(len(self.periods)):
            m_p = self.periods[i]
            t1 = TrendFilter(self.A[-4 * m_p:])
            d1 = b - t1
            r = 4 * self.m % m_p
            self.epi_S[i][r] = self.lam * d1 + (1- self.lam) * self.epi_S[i][r]
            self.K[i] = self.UpdateArray(self.K[i], self.epi_S[i][r])
            t4 = TrendFilter(self.K[i][-3 * self.m:])
            d5 = b - t1 - t4
            self.epi_T[i][r] = self.lam * d5 + (1- self.lam) * self.epi_T[i][r]
            b = b - self.epi_T[i][r]
            S.append(self.epi_T[i][r])
        self.D = self.UpdateArray(self.D, b)
        T = TrendFilter(self.D[-self.m:])
        R = x - T - sum(S)
        return T, np.array(S), R






if __name__ == '__main__':
    X = np.random.random(40)
    model = OnlineSTL(X, [7, 10])
    T, S, R = model.update(1.456)
    print('Online STL Imported')
