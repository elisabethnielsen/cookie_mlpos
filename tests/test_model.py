from cookie.model import Model
import torch
import pytest

@pytest.mark.parametrize('batch_size',[32,64])
def test_model(batch_size:int):
    model=Model()
    X=torch.randn((batch_size,1,28,28))
    Y=model(X)
    assert Y.shape == (batch_size,10) , 'Output did not match expected shape'
    
def test_value_error():
    model=Model()
    with pytest.raises(ValueError,match='Expected input to a 4D tensor'):
        model(torch.randn((1,28,28)))
    with pytest.raises(ValueError,match=r'Expected each sample to have shape \[1, 28, 28\]'):
        model(torch.randn((1,1,27,27)))
