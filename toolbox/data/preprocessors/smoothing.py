class Smoothing():
    def run(self, series):
        return series.rolling(window=30, win_type="exponential", center=True).mean().fillna(value=0)