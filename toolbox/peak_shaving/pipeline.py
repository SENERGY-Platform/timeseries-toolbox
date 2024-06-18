import logging
import sys 

import mlflow
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import kneed

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

    def determine_epsilon(self, data): # data is a collection of numbers
        logger.debug("Determine Epsilon")
        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors_fit = neighbors.fit(np.array(data).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array(data).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        logger.debug("Start KneeLocator")
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=1, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        if epsilon==0 or epsilon==None:
            return 50
        else:
            return epsilon

    def fit(self, data): # data is a collection of numbers
        logger.debug("Fit Peak Shaving")
        eps = self.determine_epsilon(data) 
        logger.debug("Start DBSCAN")
        db = DBSCAN(eps=eps, min_samples=10).fit(np.array(data).reshape(-1,1))
        clustering_labels = set(db.labels_) - {-1} # discard outlier label "-1"
        clusters = [np.array(data)[db.labels_==k] for k in clustering_labels] # This is the collection of clusters inside the data. Ouliers (i.e. label==-1) are thrwon away.
        self.min_boundaries = [min(cluster) for cluster in clusters]
        self.max_boundaries = [max(cluster) for cluster in clusters]
        logger.debug(f"Found boundaries: {self.min_boundaries},{self.max_boundaries}")