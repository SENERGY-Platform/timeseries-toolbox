from sklearn.ensemble import IsolationForest
from statistics import median
import numpy as np 

class Isolation():
    def __init__(self) -> None:
        self.all_reconstruction_errors = []

    def check(self, reconstruction_error, contam):
        # assert reconstruction_errors.shape
        anomalous_indices = []
        self.all_reconstruction_errors.append(reconstruction_error)
        anomalous_error_model = IsolationForest(contamination=contam).fit(np.array(self.all_reconstruction_errors).reshape(-1,1))
        predictions = anomalous_error_model.predict(np.array(reconstruction_error).reshape(-1,1))
        print(f"Isolation Prediction for last recon error: {predictions[-1]} - {anomalous_error_model.decision_function(np.array(reconstruction_error).reshape(-1,1))[-1]}")
        print(f"All Recon Errors: {self.all_reconstruction_errors}")
        for i in range(len(self.all_reconstruction_errors)):
            if predictions[i]==-1 and self.all_reconstruction_errors[i]>median(self.all_reconstruction_errors):
                anomalous_indices.append(i)
        return anomalous_indices

    def get_all_reconstruction_errors(self):
        return self.all_reconstruction_errors

    def set_all_reconstruction_errors(self):
        # TODO persist
        pass