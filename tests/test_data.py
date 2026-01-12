import os
from typing import Sized, cast

import pytest
import torch
from torch.utils.data import Dataset

from ml_ops.data import MyDataset, corrupt_mnist
from tests import _PATH_DATA


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)


def test_data():
    train, test = corrupt_mnist("data/processed")
    assert len(cast(Sized, train)) == 30000
    assert len(cast(Sized, test)) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(torch.tensor([y for _, y in train], dtype=torch.long))
    assert (train_targets == torch.arange(0, 10)).all()
    test_targets = torch.unique(torch.tensor([y for _, y in test], dtype=torch.long))
    assert (test_targets == torch.arange(0, 10)).all()


@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_something_about_real_data():
    # Your test code that requires the data files
    print("Data files found, running test...")
    dataset = MyDataset("data/raw")
    assert len(cast(Sized, dataset)) > 0
