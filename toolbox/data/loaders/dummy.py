from toolbox.data.loaders.loader import DataLoader

class DummyLoader(DataLoader):
    def __init__(self):
        self.data = None

    def get_data(self):
        from darts.datasets import AirPassengersDataset

        self.data = AirPassengersDataset().load()      
        df = self.data.pd_dataframe()
        df = df.reset_index()
        df = df.rename(columns={"#Passengers": "value", "Month": "time"})
        return df
