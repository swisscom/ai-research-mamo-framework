"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

AnalyticalSolver, used to solve the Quadratic
Constrained Optimization Problem for 2 gradients
analytically.

The AnalyticalSolver class contains the implementation of
the analytical QCOP Solver for 2 gradients.
"""
from copsolver.copsolver import COPSolver


class AnalyticalSolver(COPSolver):
    """AnalyticalSolver class. Inherits the COPSolver class.

    AnalyticalSolver is used to calculate the alphas for the QCOP for 2
    gradients.
    """
    def solve(self, gradients):
        """Solves the Constrained Optimization Problem for 2 gradients

            Given the gradients, compute analytically the alphas for the COP and returns them in a list of size 2.

        Args:
            gradients: numpy array of gradients of size 2 from the models
                each gradient is a numpy array of float of the same size. Gradients
                cannot be the same.Im
        Returns:
            A numpy array of floats in [0,1] of size 2 representing
            the coefficients associated to the gradients
        Raise:
            ValueError: An error occured while checking the dimensions of
                gradients
            TypeError: An error occured while accessing the argument - one
                of the arguments is NoneType
        """

        if gradients is None:
            raise TypeError('Argument: gradients type cannot be None')
        if (len(gradients) != 2):
            raise ValueError('Argument: The number of gradients must be equal to 2')
        if (len(gradients[0]) != len(gradients[1])):
            raise ValueError('Argument: The gradients must have the same length')
        if (gradients[0] == gradients[1]).all():
            return [0.5,0.5]

        r"""
        .. math::

            \alpha = \frac{(\nabla_{w}L_{2}(w) - \nabla_{w}L_{1})^{T} \star \nabla_{w}L_{2}(w)}
                          {\|\nabla_{w}L_{1} - \nabla_{w}L_{2}\|^{2}}

        Source: Multi-Gradient Descent For Multi-Objective Recommender Systems
        """

        alpha = ((gradients[1] - gradients[0]) @ gradients[1]) \
            / ((gradients[0] - gradients[1]) @ (gradients[0] - gradients[1]))

        if alpha < 0:
            alpha = 0
        if alpha > 1:
            alpha = 1

        return [alpha, 1-alpha]
