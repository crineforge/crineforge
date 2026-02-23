import pytest
from unittest.mock import patch
from crineforge.core import Trainer

@patch("crineforge.core.free_structurer")
@patch("crineforge.core.get_structurer")
@patch("crineforge.core.DataExtractor.extract", return_value="dummy extracted text")
def test_structure_stage(mock_extract, mock_get_structurer, mock_free_structurer):
    # Mocking structural JSON generations to simulate 1 data pair successfully generated
    mock_structurer_instance = mock_get_structurer.return_value
    mock_structurer_instance.generate_pairs.return_value = '[{"instruction": "test", "response": "test"}]'

    trainer = Trainer(debug_mode=True)
    trainer.load_data("dummy.txt")
    
    # Must run seamlessly without crashing while keeping structural validity intact
    trainer.structure_only()
    
    assert len(trainer.structured_pairs) > 0
