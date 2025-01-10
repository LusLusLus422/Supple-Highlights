import pytest
import spacy
import gensim

def test_spacy_installation():
    """Test if spaCy is properly installed and can load the English model."""
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp("This is a test sentence.")
        assert len(doc) > 0
    except Exception as e:
        pytest.fail(f"Failed to load spaCy model: {str(e)}")

def test_gensim_installation():
    """Test if gensim is properly installed and can create a simple model."""
    try:
        # Create a simple document for testing
        documents = [["test", "document"], ["another", "document"]]
        model = gensim.models.Word2Vec(documents, min_count=1)
        assert model is not None
    except Exception as e:
        pytest.fail(f"Failed to create gensim model: {str(e)}")

def test_text_preprocessing(sample_text):
    """Test basic text preprocessing."""
    assert isinstance(sample_text, str)
    assert len(sample_text.strip()) > 0 