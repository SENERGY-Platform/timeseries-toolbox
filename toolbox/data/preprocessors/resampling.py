class Resampler():
    def run(self, series):
        return series.resample('s').interpolate().resample('T').asfreq().dropna()