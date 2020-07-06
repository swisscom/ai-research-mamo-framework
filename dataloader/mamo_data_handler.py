"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Abstract Mamo Data Handler class, main source of feeding data
for the MAMO framework.

This is the main class that will supply data to the MAMO framework.
Users that need custom Data Loaders (ex. for multitask learning),
need to create implementation of this class that suits their need.
The class consists only of abstract methods giving the users a basic
skeleton and a freedom to define custom Data Handlers and load data
from different sources. We include an implementation of this class
for developing Auto Encoder models in the 'ae_data_handler.py' script.

"""

from abc import ABC, abstractmethod


class MamoDataHandler(ABC):
    """Abstract Mamo Data Handler class.

    The Mamo Data Handler class lists all the methods that need to
    be implemented by any Data Handler to feed data to the framework.

    Attributes:
        _dataset_name: A string indicating the name of the dataset (ex. MovieLens).
    """

    def __init__(self, dataset_name):
        """Inits the Mamo Data Handler with dataset name.

        Args:
            dataset_name: A string indicating the name of the dataset (ex. MovieLens).
        """
        self._dataset_name = dataset_name

    def get_name(self):
        """Returns the name of the dataset that this data handler handels."""
        return self._dataset_name

    @abstractmethod
    def get_train_dataloader(self, batch_size=500, shuffle=True, drop_last=True, **other):
        """Returns a pytorch DataLoader for the training dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the training dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.
            other: Directory for passing custom arguments, for full flexibility.

        Returns:
            Returns pytorch DataLoader object.
        """
        pass

    @abstractmethod
    def get_validation_dataloader(self, batch_size=500, shuffle=True, drop_last=True, **other):
        """Returns a pytorch DataLoader for the validating dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the validating dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.
            other: Directory for passing custom arguments, for full flexibility.

        Returns:
            Returns pytorch DataLoader object.
        """
        pass

    @abstractmethod
    def get_test_dataloader(self, batch_size=500, shuffle=True, drop_last=True, **other):
        """Returns a pytorch DataLoader for the testing dataset.

        A DataLoader represents a Python iterable over a dataset with
        additional functions like batching, shuffling of the data, etc.
        This function creates and returns a DataLoader created on
        the testing dataset.

        Args:
            batch_size: Integer, how many samples per batch to load, default=500.
            shuffle: Boolean, set to True to have the data reshuffled at every epoch, default=True.
            drop_last: Boolean, set to True to drop the last incomplete batch, default=True.
            other: Directory for passing custom arguments, for full flexibility.

        Returns:
            Returns pytorch DataLoader object.
        """
        pass

    @abstractmethod
    def get_traindata_len(self):
        """Returns the number of samples in the training dataset.

        Returns:
            Returns integer, the number of samples in the
            training dataset.
        """
        pass

    @abstractmethod
    def get_validationdata_len(self):
        """Returns the number of samples in the validating dataset.

        Returns:
            Returns integer, the number of samples in the
            validating dataset.
        """
        pass

    @abstractmethod
    def get_testdata_len(self):
        """Returns the number of samples in the testing dataset.

        Returns:
            Returns integer, the number of samples in the
            testing dataset.
        """
        pass

    @abstractmethod
    def get_input_dim(self):
        """Returns the second dimension of the input data.

        Returns:
            Returns integer, the second dimension of the input data.
        """
        pass

    @abstractmethod
    def get_output_dim(self):
        """Returns the second dimension of the output data.

        Returns:
            Returns integer, the second dimension of the output data.
        """
        pass
