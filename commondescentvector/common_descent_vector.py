"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Abstract CommonDescentVector class, used for defining a Common Descent Vector

The abstract CommonDescentVector class contains a basic skeleton and some implementation
details that will be shared among its children.
"""
from abc import ABC, abstractmethod


class CommonDescentVector(ABC):
    """Abstract class for Common Descent Vector.

    Defines the methods needed for instantiating a Common Descent Vector
    """

    @abstractmethod
    def get_descent_vector(self, losses, gradients):
        """Compute and return the Common Descent Vector

        Returns:
            A numpy array of PyTorch tensor representing the losses
        Raises:
            TypeError: An error occured when attributes are not set.
        """
        pass
