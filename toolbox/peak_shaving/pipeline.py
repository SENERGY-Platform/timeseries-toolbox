import logging
import sys 

import mlflow
import numpy as np
from KDEpy import FFTKDE
from scipy.signal import argrelextrema
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

class PeakShavingPipeline(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, context, data, params=None):  
        pass 

    def get_cluster_min_boundaries(self):
        return self.min_boundaries

    def get_cluster_max_boundaries(self):
        return self.max_boundaries

    def fit(self, data): # data is a collection of numbers
        logger.debug("Fit Peak Shaving")
        logger.debug("Start Clustering")
        x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
        local_minima = list(x[argrelextrema(y, np.less)[0]])
        self.min_boundaries = [0]+local_minima
        self.max_boundaries = local_minima+[max(data)]
        logger.debug(f"Found boundaries: {self.min_boundaries},{self.max_boundaries}")

   
