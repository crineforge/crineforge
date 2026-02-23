import pytest
from unittest.mock import patch
from crineforge.model.gpu import GPUSensitive

@patch("torch.cuda.is_available", return_value=False)
def test_cpu_fallback(mock_cuda):
    strategy = GPUSensitive.get_strategy()
    assert strategy["device"] == "cpu"
    assert strategy["precision"] == "float32"
