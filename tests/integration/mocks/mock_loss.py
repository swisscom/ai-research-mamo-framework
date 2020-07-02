from loss.loss_class import Loss
import torch

# I will make this into a real loss with tests soon but right now I just need
# it like this.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MSELoss(Loss):
    def __init__(self):
        """Initialize the loss."""
        super().__init__('MSELoss')

    def compute_loss(self, y_true, output_model):
        output_model = output_model.to(device)
        return ((output_model - y_true) ** 2).sum()\
            / output_model.data.nelement()
