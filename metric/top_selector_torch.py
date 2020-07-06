"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

TopSelectorTorch class, used selecting top samples.

The TopSelectorTorch class is a helper class used by different metric
implementations to select the top values from arrays. This logic is extracted
into a class of its own to increase reusability and simplify testing.
This class uses PyTorch and GPU resources when possible.
"""
import torch


class TopSelectorTorch:
    """TopSelectorTorch class."""

    def find_top_k_binary(self, values, k):
        """Finds the top k values for each row of a matrix and returns a binary
        mask on their positions.

        The method masks the k input values with the highest numerical value in
        every row of the input 2D numpy array.

        Args:
            values: A PyTorch tensor of values.
            k: An integer that denotes the number of values to obtain from the
                ranking of the values. The method masks the k values with the
                highest scores.

        Returns:
            A binary mask in the form of a 2D Pytorch tensor that outputs the
            top k values per row from the input values.
            For example:

            values = tensor([[0.5, 0.7, 0.3],
                                 [0.4, 0.1, 0.7]])
            k = 2
            find_top_k_binary returns:

            tensor([[ True, True, False],
                   [ True, False, True]])

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                values argument. It is not a 2D tensor. Or if k is smaller
                than 0.
        """
        if values is None:
            raise TypeError('Argument: values must be set.')
        if k is None:
            raise TypeError('Argument: k must be set.')
        if not isinstance(k, int):
            raise TypeError('Argument: k must be an integer.')
        if not torch.is_tensor(values):
            raise TypeError('Argument: values must be a PyTorch tensor.')
        if values.ndimension() != 2:
            raise ValueError('Argument: values must be a 2D tensor.')
        if k < 1:
            raise ValueError('Argument: k cannot be negative.')
        if k >= values.shape[1]:
            raise ValueError('Argument: k cannot be larger than\
                             values.shape[1]')

        _, idx = torch.topk(values, k=k, dim=1, sorted=False)
        values_binary = torch.zeros_like(values, dtype=torch.bool)
        values_binary = values_binary.scatter(1, idx[:, :k], True)
        values_binary[values <= 0] = False
        return values_binary
