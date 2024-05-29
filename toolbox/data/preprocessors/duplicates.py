class Duplicates():
    # Remove duplicate timestamps
    def run(self, series):
        return series[~series.index.duplicated(keep='first')]