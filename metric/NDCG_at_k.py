"""NDCGAtK, used for calculating the Normalized Discounted Cumulative Gain.

The NDCGAtK class contains the implementation of the NDCG metric.
Its function is to evaluate results obtained using a certain model.
"""
import math
import numpy as np
from scipy.stats import rankdata
from metric.metric_at_k import MetricAtK


class NDCGAtK(MetricAtK):
    """NDCGAtK class. Inherits the MetricAtK class.

    The NDCGAtK is used to calculate the Normalized Discounted Cumulative Gain
    metric.
    """
    def __init__(self, k):
        """Inits NDCGAtK with its k value.
        k must be greater than 0.
        Raises:
            TypeError: The k value is not an integer or is not set.
            ValueError: The k value is smaller than 1.
        """
        super().__init__('NDCG', k)

    def evaluate(self, y_true, y_pred):
        """Evaluates the given predictions with the NDCG metric.

        Calculates the NDCG on the passed predicted and true values at k.

        Args:
            y_true: A PyTorch tensor of true values. Only one value per row
                    can be > 0!
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Will return a float with the calculated NDCG value.
            The NDCG definition is:
            math::
            NDCG@K = Z_k \\sum_{i=1}^{K} \\frac{2^r_i - 1}{log_2 (i + 1)}
            r_i = 1 if in true set, 0 otherwise
            From:
            https://www.comp.nus.edu.sg/~kanmy/papers/cikm15-trirank-cr.pdf

        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments. One or both are not a 2D arrays,
                or they are 2D but of different sizes along those dimensions.
                If y_true has more than one true value per row this error
                is raised. This is also raised if the output is not in [0,1].
        """
        self._check_input(y_true, y_pred)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        self._check_args_numpy(y_pred, y_true)
        # Check only one ground truth value = 1 per row in y_true.
        y_true[y_true > 0] = 1
        y_true[y_true < 0] = 0
        for x in np.sum(y_true, axis=1):
            if x != 1:
                raise ValueError('Incorrect format of argument: y_true. \
                                  Input must have only one true value \
                                  per row.')
        rows = y_pred.shape[0]
        cols = y_pred.shape[1]
        NDCG = 0
        for i in range(rows):
            rank = rankdata(-y_pred[i], method='min')
            rank[rank > self._k] = 0
            for j in range(cols):
                if y_pred[i][j] <= 0 or rank[j] < 1:
                    continue
                NDCG += (2**y_true[i][j] - 1)\
                    / math.log(rank[j]+1, 2)
        result = NDCG / rows
        if not (0 <= result <= 1):
            raise ValueError('The output of NDCGAtK.evaluate must be in [0,1]')
        return result
