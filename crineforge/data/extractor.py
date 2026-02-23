import os
import json
import pandas as pd
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ..utils.logger import get_logger

logger = get_logger(__name__)

MAX_FILE_MB = 10

class DataExtractor:
    """Extracts raw text from multiple document formats."""
    
    @staticmethod
    def extract(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_MB:
            raise ValueError(f"File size guard triggered. File is {file_size_mb:.2f}MB, limit is {MAX_FILE_MB}MB.")
            
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        logger.info(f"Extracting data from {ext} file: {file_path}")
        
        if ext == ".pdf":
            return DataExtractor._extract_pdf(file_path)
        elif ext == ".csv":
            return DataExtractor._extract_csv(file_path)
        elif ext == ".json":
            return DataExtractor._extract_json(file_path)
        elif ext in [".txt", ".md"]:
            return DataExtractor._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        if fitz is None:
            raise ImportError("PyMuPDF is not installed. Run `pip install PyMuPDF`.")
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text

    @staticmethod
    def _extract_csv(file_path: str) -> str:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)

    @staticmethod
    def _extract_json(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    @staticmethod
    def _extract_txt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
