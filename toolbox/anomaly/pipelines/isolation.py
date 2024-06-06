from sklearn.ensemble import IsolationForest
from statistics import median
import numpy as np 

from .utils import Reconstruction

class Isolation():
    def __init__(self, time_to_keep_values) -> None:
        self.all_reconstruction_errors: list[Reconstruction] = []
        self.time_to_keep_values = time_to_keep_values

    def check(self, current_reconstruction_error: Reconstruction, contam):
        # assert reconstruction_errors.shape
        anomalous_indices = []
        self.all_reconstruction_errors.append(current_reconstruction_error)
        self.discard_old_errors()

        all_reconstruction_error_values = np.array([recon.reconstruction_error for recon in self.all_reconstruction_errors]).reshape(-1,1)
        current_reconstruction_error_value = np.array(current_reconstruction_error.reconstruction_error).reshape(-1,1)
        
        anomalous_error_model = IsolationForest(contamination=contam).fit(all_reconstruction_error_values)
        predictions = anomalous_error_model.predict(current_reconstruction_error_value)
        print(f"Isolation Prediction for last recon error: {predictions[-1]} - {anomalous_error_model.decision_function(current_reconstruction_error_value)[-1]}")
        print(f"All Recon Errors: {all_reconstruction_error_values}")
        for i in range(len(all_reconstruction_error_values)):
            if predictions[i]==-1 and all_reconstruction_error_values[i]>median(all_reconstruction_error_values):
                anomalous_indices.append(i)
        return anomalous_indices

    def get_all_reconstruction_errors(self):
        return self.all_reconstruction_errors

    def set_all_reconstruction_errors(self, all_reconstruction_errors):
        self.all_reconstruction_errors = all_reconstruction_errors

    def discard_old_errors(self):
        last_timestamp = self.all_reconstruction_errors[-1].timestamp
        self.all_reconstruction_errors = list(filter(lambda error: (last_timestamp - error.timestamp) <= self.time_to_keep_values, self.all_reconstruction_errors))