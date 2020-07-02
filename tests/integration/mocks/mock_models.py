
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MockNoChange(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockNoChange, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns the input.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        return input.clone().detach().to(device)


class MockAllZeros(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockAllZeros, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns all zeros.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        return torch.zeros(input.size(), dtype=torch.double).to(device)


class MockOpposite(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockOpposite, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns input - 1.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        output = input.clone().detach()
        output[input == 1] = 0
        output[input == 0] = 1
        return output.to(device)


class MockShiftRightByOne(nn.Module):
    def __init__(self):
        """Initialize the model"""
        super(MockShiftRightByOne, self).__init__()

    def forward(self, input):
        """A single forward pass of the model. Returns input shifted to the
        right by one along axis 1.

        Args:
            input: The input to the model as a tensor of
                batch_size X input_size
        """
        tmp = input.cpu().numpy()
        tmp = np.roll(tmp, shift=1, axis=1)
        return torch.from_numpy(tmp).to(device)
