from sklearn.ensemble import IsolationForest
from statistics import median
import numpy as np 

class Isolation():
    def __init__(self) -> None:
        self.all_reconstruction_errors = np.array()

    def check(self, reconstruction_errors, contam):
        # reconstruction_errors: 1d
        # reconstruction_errors all errors within the batch
        anomalous_indices = []
        anomalous_error_model = IsolationForest(contamination=contam).fit(np.array(self.all_reconstruction_errors).reshape(-1,1))
        predictions = anomalous_error_model.predict(np.array(reconstruction_errors).reshape(-1,1))
        for i in range(len(reconstruction_errors)):
            if predictions[i]==-1 and reconstruction_errors[i]>median(reconstruction_errors):
                anomalous_indices.append(i)
        self.all_reconstruction_errors = np.vstack(reconstruction_errors)
        return anomalous_indices