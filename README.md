<a href="https://github.com/SENERGY-Platform/timeseries-toolbox/actions/workflows/tests.yaml" rel="nofollow">
    <img src="https://github.com/SENERGY-Platform/timeseries-toolbox/actions/workflows/tests.yaml/badge.svg" alt="Tests" />
</a>

# TimeSeries Toolbox

## Forecasting 
Load a specific Model/Pipeline
```python
from toolbox.estimation.load import get_pipeline

ProphetPipeline = get_pipeline("DartProphet")
pipeline = ProphetPipeline(freq="H", add_time_covariates=False, )
```

Each model implements a `fit`, `predict` and `get_hyperparams` method.

## Structure
`model_selection` and `parameter_tuning` are not used right now.