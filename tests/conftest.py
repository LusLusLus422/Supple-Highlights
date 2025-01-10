import pytest
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_text():
    """Fixture providing sample text for testing."""
    return """
    This is a sample text with some highlights.
    It contains multiple sentences and paragraphs.
    This will be used for testing text processing functions.
    """

@pytest.fixture
def sample_pdf_path():
    """Fixture providing path to a sample PDF file for testing."""
    return os.path.join(os.path.dirname(__file__), 'data', 'sample.pdf')

@pytest.fixture
def test_db_url():
    """Fixture providing a test database URL."""
    return 'sqlite:///:memory:' 