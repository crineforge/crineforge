import pytest
from crineforge.data.extractor import DataExtractor

def test_txt_extraction(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("Hello CrineForge")

    result = DataExtractor.extract(str(file))
    assert "Hello CrineForge" in result
