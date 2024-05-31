import torch 

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