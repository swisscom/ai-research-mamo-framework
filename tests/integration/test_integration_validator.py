from metric.recall_at_k import RecallAtK
from dataloader.mamo_dataset import MamoDataset
from validator import Validator
from torch.utils.data import DataLoader
from tests.integration.mocks.mock_models import MockAllZeros
from tests.integration.mocks.mock_models import MockNoChange
from tests.integration.mocks.mock_models import MockOpposite
from tests.integration.mocks.mock_models import MockShiftRightByOne
from tests.integration.mocks.mock_loss import MSELoss
import numpy as np

# Packages needed to run test:
# os
# numpy
# torch
# pytest

# Variables
# Mock dataset
input_data = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
input_data = input_data.astype(float)


# Test demonstrating mock all zeros model to show missing
# metrics or missing objectives
def test_validator_mock_all_zeros_model():
    mock_dataset = MamoDataset(input_data, input_data.copy())
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_all_zeros = Validator(MockAllZeros(), mock_dataloader,
                            [RecallAtK(1)], None)
    results = v_all_zeros.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 0)
    assert isinstance(results[1], list)
    assert(results[1] == [])
    v_all_zeros = Validator(MockAllZeros(), mock_dataloader,
                            None, [MSELoss()])
    results = v_all_zeros.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0] == [])
    assert isinstance(results[1], list)
    mse = np.mean(input_data)
    assert(round(results[1][0], 2) == round(mse, 2))
    assert(round(v_all_zeros.combine_objectives(results[1]), 2)
           == round(mse, 2))


# Test demonstrating mock no change model
# Recall is 0 as we are recommending already chosen elements
def test_validator_mock_no_change_model():
    mock_dataset = MamoDataset(input_data, input_data.copy())
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_no_change = Validator(MockNoChange(), mock_dataloader,
                            [RecallAtK(1)], [MSELoss()])
    results = v_no_change.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 0)
    assert isinstance(results[1], list)
    # Removing chosen elements -so:
    mse = np.mean(input_data)
    assert(round(results[1][0], 2) == round(mse, 2))
    assert(round(v_no_change.combine_objectives(results[1]), 2)
           == round(mse, 2))


# Test demonstrating mock opposite model
def test_validator_mock_opposite_model():
    mock_dataset = MamoDataset(input_data, input_data.copy())
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_opposite = Validator(MockOpposite(), mock_dataloader,
                           [RecallAtK(1)], [MSELoss()])
    results = v_opposite.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 0)
    assert isinstance(results[1], list)
    assert(results[1][0] == 1)
    assert(v_opposite.combine_objectives(results[1]) == 1)


# Test demonstrating mock shift right by one model
def test_validator_mock_shift_right_by_one_model():
    mock_dataset = MamoDataset(input_data, np.roll(input_data.copy(),
                               shift=1, axis=1))
    mock_dataloader = DataLoader(mock_dataset, batch_size=1,
                                 shuffle=False, drop_last=False)
    v_shift_right = Validator(MockShiftRightByOne(), mock_dataloader,
                              [RecallAtK(1)], [MSELoss()])
    results = v_shift_right.evaluate()
    assert isinstance(results, tuple)
    assert isinstance(results[0], list)
    assert(results[0][0] == 1)
    assert isinstance(results[1], list)
    assert(results[1][0] == 0)
    assert(v_shift_right.combine_objectives(results[1]) == 0)
