import torch
import pytest

from cookie.data import corrupt_mnist

from tests import _PATH_DATA

import os.path
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason=f"Data files not found at {_PATH_DATA}")
def test_data():
    train,test = corrupt_mnist()
    assert len(train)==30000 ,"Dataset did not have the correct number of samples"
    assert len(test)==5000 , "Dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28) , 'Image did not have the correct shape (1,28,28)'
            assert y in range(10) ,'Label numbers out of range (0-9)' # because 10 classes 
    #that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    #assert that all labels are represented
        targets=torch.unique(dataset.tensors[1])
        assert torch.equal(targets,torch.tensor(range(10))) , 'Targets did not contain all labels'