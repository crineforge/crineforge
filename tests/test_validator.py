import pytest
from crineforge.data.validator import Validator

def test_dataset_guard_raises():
    data = [{"instruction": "a", "output": "b"}]
    with pytest.raises(ValueError):
        Validator.validate_dataset_size(data, debug_mode=False)

def test_dataset_guard_debug_allows():
    data = [{"instruction": "a", "output": "b"}]
    assert Validator.validate_dataset_size(data, debug_mode=True)
