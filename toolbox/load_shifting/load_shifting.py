import logging 

from ray import air, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ray.air import session

from toolbox.ml_config import Config

class LoadShifting():
    def __init__(self, mlflow_url) -> None:
        super().__init__()
        self.mlflow_url = mlflow_url

    '''Glätten, Shiften, Padden'''

    def smoothen(self, load_curve, window_length):# Berechnet exponentiale Glättung einer Lastkurve (In- und Output als array). Glättung ist notwendig, um später Optimierungs-Algorithmen anwenden zu können.
        window_length = window_length #Parameter für die Glättung. Muss für jede Kurve einzeln gewählt werden.
        df = pd.DataFrame(load_curve)
        return np.array(df.rolling(window=int(window_length), win_type="exponential", center=True).mean().fillna(value=0)).flatten()

    def round_shifts(self, shift_lengths, decimals=2): # Rundet die Shift-Werte auf Vielfache der resolution, damit die Lastgangkurven nach den Shifts weiter auf den gleichen Stützstellen definiert sind.
        return np.array(shift_lengths).round(decimals=decimals).flatten()

    def end_pad_zeros(self, list_of_shifted_load_curves): # Padded die geshifteten Lastgang-Kurven, damit nach dem Shiften wieder alle Kurven die gleiche Länge haben.
        aux_list = []
        max_length = max([len(shifted_load_curve) for shifted_load_curve in list_of_shifted_load_curves])
        for shifted_load_curve in list_of_shifted_load_curves:
            padded_shifted_load_curve = np.hstack((shifted_load_curve, np.zeros(max_length-len(shifted_load_curve))))
            aux_list.append(padded_shifted_load_curve)
        return np.vstack(aux_list)

    def shift_loads(self, array_of_loads, resolution, *shift_lengths):
        rounded_shift_lengths = self.round_shifts(shift_lengths)
        aux_list = []
        for i in range(len(rounded_shift_lengths)):
            load_curve = array_of_loads[i]
            if rounded_shift_lengths[i] >= 0:
                start_zeros = np.zeros(int(rounded_shift_lengths[i]*(1/resolution)))
                shifted_load = np.hstack((start_zeros, load_curve))
            else:
                shifted_load = load_curve[-int(rounded_shift_lengths[i]*(1/resolution)):]
            aux_list.append(shifted_load)
        padded_shifted_loads_array = self.end_pad_zeros(aux_list)
        return padded_shifted_loads_array
        

    '''Aggregieren der geshifteten Lastgang-Kurven und Max-Berechnung'''

    def load_sum(self, array):# Berechnet die Summe der Kurven, die als Zeilen in array abgelegt sind.
        return np.sum(array, axis=0)

    def compute_load_peak(self, curve):# Berechnet das Maximum einer Kurve (gegeben als array). Das ist keine komplexe Berechnung. Hier kann einfach das Maximum über das Array berechnet werden.
        return np.max(curve)

    def max_of_sum(self, array):# Berechnet das Maximum der Summe der Kurven, die als Zeilen in array abgelegt sind.
        combined_loads = self.load_sum(array)
        return self.compute_load_peak(combined_loads)

    '''Berechnung der target function für die Minimierung (berechnet wird der maximale Peak der Last-Aggregation der geshifteten Kurven in Abhängigkeit der Shift-Weiten)'''

    def target_function(self, array_of_loads, resolution, smoothing_window_lengths, *shift_lengths):
        array_of_smoothened_load_curves = np.vstack([self.smoothen(array_of_loads[i], smoothing_window_lengths[i]) for i in range(array_of_loads.shape[0])]) 
        # Berechnet die Glätungen der einzelne Last-Kurven, die als Zeilen in array_of_loads abgelegt sind.  
        padded_shifted_loads = self.shift_loads(array_of_smoothened_load_curves, resolution, *shift_lengths)
        # Berechnet die Shifts der geglätteten Kurven und padded mit Nullen am Ende
        return self.max_of_sum(padded_shifted_loads)# Berechnet das Maximum der Summe der geshifteten Kurven.

    '''Approximimerte Lösung des Minimierungs-Problems'''

    def find_optim_shift_loads_per_config(self, loads, resolution, smoothing_window_lengths, parameter_config):
        print("Run optimization")
        OptimizeResult = minimize(lambda x: self.target_function(loads, resolution, smoothing_window_lengths, *x), parameter_config['x0'], method='Powell', options={'xtol': parameter_config['xtol'], 'ftol': parameter_config['ftol']})
        max_sum = self.target_function(loads, resolution, smoothing_window_lengths, *(OptimizeResult.x))
        return OptimizeResult.x, max_sum

    def tune_with_param(self, parameter_config, loads, resolution, smoothing_window_lengths):
        _, max_load_sum = self.find_optim_shift_loads_per_config(loads, resolution, smoothing_window_lengths, parameter_config)
        session.report({
            "max_load_sum": max_load_sum
        })

    def run_across_hyperparams(self, hyperparams, experiment_name, loads, resolution, smoothing_window_lengths):
        # Define a tuner object.
        tuner = tune.Tuner(
                tune.with_parameters(self.tune_with_param, loads=loads, resolution=resolution, smoothing_window_lengths=smoothing_window_lengths),
                param_space=hyperparams,
                run_config=air.RunConfig(
                    name="tune_model",
                    # Set Ray Tune verbosity. Print summary table only with levels 2 or 3.
                    verbose=2,
                    callbacks=[MLflowLoggerCallback(
                        tracking_uri=self.mlflow_url,
                        experiment_name=experiment_name,
                        save_artifact=True,
                    )]
                )
        )

        # Fit the tuner object.
        results = tuner.fit()
        return results

    def run_load_shifting(self, preprocessed_data):
        '''Parameter, die während des Pre-Processings der einzelnen Lastgangkurven bestimmt werden müssen'''
        config = Config()
        logging.info("RUN LOAD SHIFTING")
        logging.debug(f"CONFIG: {config}")
        loads_matrix = preprocessed_data['loads']
        resolution = preprocessed_data['resolution']
        smoothing_window_lengths = preprocessed_data['smoothing_window_lengths']

        # Anzahl der verschiedenen Lastgangkurven.
        number_of_loads = loads_matrix.shape[0]

        hyperparams = {
            'xtol': 0.0000001,
            'ftol': 0.0000001,
            'x0': tune.grid_search([np.random.uniform(-5, 5, size=(number_of_loads,)) for _ in range(number_of_loads)]) # Startpunkt des Minimierungs-Algorithmus. Wichtigster Hyper-Paramter.
        }
        tuning_result = self.run_across_hyperparams(hyperparams, config.EXPERIMENT_NAME, loads_matrix, resolution, smoothing_window_lengths)
        best_result = tuning_result.get_best_result("max_load_sum", "min")
        best_config = best_result.config
        print(f"Best Parameters: {best_config}")

        optimal_shifted_loads, _ = self.find_optim_shift_loads_per_config(loads_matrix, resolution, smoothing_window_lengths, best_config)
        print(optimal_shifted_loads)
        
    def run(self, data):
        self.run_load_shifting(data)