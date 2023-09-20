from darts.dataprocessing.transformers.scaler import Scaler 
from gluonts.dataset.pandas import PandasDataset
from darts import TimeSeries

def create_darts_encoder_based_on_freq(freq):
    date_covariates = ["month"]

    if freq == 'H':
        date_covariates.append("hour")
    elif freq == 'D':
        date_covariates.append("dayofweek")

    encoders = {
            "datetime_attribute": {"future": date_covariates},
            #"position": {"past": ["relative"], "future": ["relative"]}, TODO absolute past future?
            "transformer": Scaler(),
    }

    return encoders


def convert_to_gluon_pandas_dataset(df, target_column_name):
    dataset = PandasDataset.from_long_dataframe(df, target=target_column_name) # TODO covariates 
    return dataset

def convert_df_to_ts(df, time_col="time", value_cols="value"):
    ts = TimeSeries.from_dataframe(df, time_col=time_col, value_cols=value_cols)
    return ts