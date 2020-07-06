"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Abstract COPSolver class, used for defining a Constrained Optimization
Problem Solver.

The Abstract COPSolver class contains a basic skeleton and some implementation
details that will be shared among its children.
"""
from abc import ABC, abstractmethod


class COPSolver(ABC):
    """Abtract COPSolver class

    The Abstract COPSolver class represents the parent class that is
    inherited if the solve method is correctly implemented
    """

    @abstractmethod
    def solve(self, gradients):
        """Solves the Constrained Optimization Problem

            Given the gradients, compute the alphas for the COP and returns
            them in a list.

        Args:
            gradients: A list of gradients from the models
        Returns:
            Will return a list of coefficients associated to the gradients
        Raises:
            ValueError: An error occured when checking the dimensions of
            the gradients argument.
        """
        return
