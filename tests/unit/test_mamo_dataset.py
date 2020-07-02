"""This test doesn't need any custom library or any data loading.
To run them just execute 'pytest'.
"""
from dataloader.mamo_dataset import MamoDataset
import pytest
import numpy as np

# Tests for 1-d data
test_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_output = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])

# Two objects for testing
mamoDataset1 = MamoDataset(test_input)
mamoDataset2 = MamoDataset(test_input, test_output)


# Testing the none input data error
def test_none_value_error_1d():
    input_data = None
    output_data = np.array([-1, -2])
    with pytest.raises(ValueError, match='The input data is None, please give a valid input data.'):
        mamoDataset_exception = MamoDataset(input_data, output_data)
        mamoDataset_exception.__len__()

# Testing the length error


def test_length_value_error_1d():
    input_data = np.array([1, 2, 3])
    output_data = np.array([-1, -2])
    with pytest.raises(ValueError, match='The length of the input data must match the length of the output data!'):
        mamoDataset_exception = MamoDataset(input_data, output_data)
        mamoDataset_exception.__len__()


# Testing the length method
def test_len_1d():
    assert len(test_input) == mamoDataset1.__len__()
    assert len(test_input) == mamoDataset2.__len__()
    assert len(test_output) == mamoDataset2.__len__()


# Testing the getitem method
def test_get_item_1d():
    # only input
    x, y = mamoDataset1.__getitem__(0)
    assert x == y == 1
    x, y = mamoDataset1.__getitem__(5)
    assert x == y == 6
    # input and output
    x, y = mamoDataset2.__getitem__(0)
    assert x == 1
    assert y == -1
    x, y = mamoDataset2.__getitem__(5)
    assert x == 6
    assert y == -6
    x, y = mamoDataset2.__getitem__(-5)
    assert x == 6
    assert y == -6


# Testing the none input data error
def test_get_item_1d_errors():
    with pytest.raises(IndexError, match=r'index [0-9]+ is out of bounds for dimension 0 with size [0-9]+'):
        x, y = mamoDataset1.__getitem__(55)
    with pytest.raises(IndexError):
        x, y = mamoDataset1.__getitem__(5.5)


# Tests for 2-d data
test_input_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [
    9, 10], [11, 12], [13, 14], [15, 16]])
test_output_2d = np.array([[-1, -2], [-3, -4], [-5, -6],
                           [-7, -8], [-9, -10], [-11, -12], [-13, -14], [-15, -16]])


mamoDataset1_2d = MamoDataset(test_input_2d)
mamoDataset2_2d = MamoDataset(test_input_2d, test_output_2d)


# Testing the length error
def test_length_value_error_2d():
    input_data = np.array([[1, 2], [3, 4], [5, 6]])
    output_data = np.array([-1, -2])
    with pytest.raises(ValueError, match='The length of the input data must match the length of the output data!'):
        mamoDataset_exception = MamoDataset(input_data, output_data)
        mamoDataset_exception.__len__()


# Testing the length method
def test_len_2d():
    assert len(test_input_2d) == mamoDataset1_2d.__len__()
    assert len(test_input_2d) == mamoDataset2_2d.__len__()
    assert len(test_output_2d) == mamoDataset2_2d.__len__()


# Testing the getitem method
def test_get_item_2d():
    # only input
    x, y = mamoDataset1_2d.__getitem__(0)
    assert x[0] == y[0] == 1
    assert x[1] == y[1] == 2
    x, y = mamoDataset1_2d.__getitem__(5)
    assert x[0] == y[0] == 11
    assert x[1] == y[1] == 12
    # input and output
    x, y = mamoDataset2_2d.__getitem__(0)
    assert x[0] == 1
    assert x[1] == 2
    assert y[0] == -1
    assert y[1] == -2
    x, y = mamoDataset2_2d.__getitem__(5)
    assert x[0] == 11
    assert x[1] == 12
    assert y[0] == -11
    assert y[1] == -12
