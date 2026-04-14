import torch


class CreditSpectral:

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, x, edge_index):

        row, col = edge_index

        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv = 1.0 / (deg + 1e-6)

        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])

        x_smooth = deg_inv.unsqueeze(1) * agg

        return (1 - self.alpha) * x + self.alpha * x_smooth