import torch 

QUANTILS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

class Quantil():
    def __init__(self, train_losses) -> None:
        self.train_losses = train_losses

    def calc_threshold(self, losses, quantil):
        quantiles = torch.tensor([quantil], dtype=torch.float32)
        quants = torch.quantile(losses, quantiles)
        threshold = quants[0]
        return threshold

    def check(self, quantil, reconstruction_errors):
        threshold = self.calc_threshold(self.train_losses, quantil)
        anomaly_indices = torch.where(reconstruction_errors > threshold)[0]
        normal_indices = torch.where(reconstruction_errors < threshold)[0]
            
        return anomaly_indices, normal_indices
    
    def _find_best_quantil(self, pipeline, test_data):
        # TODO: not needed really
        # Quantils are also parameters but not for training
        best_quantil = None
        results_per_quantil = {}
        best_loss = None
        for quantil in QUANTILS:
            reconstructions, anomaly_indices, normal_indices, test_losses = pipeline.predict_with_quantil(test_data, quantil)
            results_per_quantil[quantil] = {
                "reconstructions": reconstructions,
                "anomaly_indices": anomaly_indices,
                "normal_indices": normal_indices,
                "test_losses": test_losses
            }

            loss = test_losses.sum().item()

            if best_loss == None:
                best_loss = loss 
                best_quantil = quantil
            elif loss < best_loss:
                best_loss = loss
                best_quantil = quantil

        metrics = {
            "loss": best_loss
        }

        pipeline.set_quantil(best_quantil)

        # Generate plots
        plots = []
        reconstructions_of_best_quantil = results_per_quantil[best_quantil]['reconstructions']
        normal_indices_of_best_quantil =  results_per_quantil[best_quantil]['normal_indices']
        anomaly_indices_of_best_quantil =  results_per_quantil[best_quantil]['anomaly_indices']
        
        if len(reconstructions) > 0:
            if len(normal_indices_of_best_quantil) > 0:
                normal_recons_plot = plot_reconstructions(reconstructions_of_best_quantil, normal_indices_of_best_quantil, test_data, "Normal")
                plots.append(normal_recons_plot)

            if len(anomaly_indices_of_best_quantil) > 0:
                anomaly_recons_plot = plot_reconstructions(reconstructions_of_best_quantil, anomaly_indices_of_best_quantil, test_data, "Anomaly")
                plots.append(anomaly_recons_plot)

        losses_of_best_quantil = results_per_quantil[best_quantil]['test_losses']
        losses_hist = plot_losses(losses_of_best_quantil)
        plots.append(losses_hist)

        return pipeline, metrics, plots
