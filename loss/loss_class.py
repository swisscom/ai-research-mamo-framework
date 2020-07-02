"""Loss Class for all the losses of the MAMO framework.
  Typical usage example:  foo = Loss()
  bar = foo.compute_loss()
"""

from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Loss class that will be specific for models / datasets / goals.
    This class handles the computation of the loss functions for the MAMO framework.

    Attributes:
        name: name of the loss.
    """

    def __init__(self, name):
        """
        Inits LossClass with a name.
        Example: for VAE losses can be the Reconstruction or Regularization.
        """
        self.name = name

    def __check_dim_pred_gt__(self, y_true, y_pred):
        """
        Function that checks if the shape of y_pred == y_true, return an error
        """
        if(y_pred.shape != y_true.shape):
            raise ValueError(
                'The dimensions of predictions (y_pred) and ground-truth (y_true) should be the same,'
                + ' got {} and {}.'.format(y_pred.shape, y_true.shape))

    def __check_is_mean_var__(self, mean, log_variance):
        """
        Function that checks that mean or var are not None
        """
        if(mean is None):
            raise Exception('mean should not be None')
        elif(log_variance is None):
            raise Exception('var should not be None')

    @abstractmethod
    def compute_loss(self, y_true, output_model):
        """
        This method compute the loss from predictions and ground-truth.
        abstract method that needs to be defined by every Loss submodule.

         Args:
            y_true: ground-truth labels.
            output_model: specific output for the model, for example:
                          output_model = (y_pred, mean, logvar) where
                          y_pred are the predictions (of the model),
                          mean and logvar are related to the model in the case of VAE.

        Returns:
            The actual loss (float)
        """
        pass
