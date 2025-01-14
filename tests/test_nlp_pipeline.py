"""Test suite for NLP pipeline functionality."""

import unittest
import os
import tempfile
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.models.models import Document, Highlight, Topic, WatchedDirectory
from app.models.base import Base
from app.nlp_pipeline.pipeline import NLPPipeline

class TestNLPPipeline(unittest.TestCase):
    """Test cases for NLP pipeline functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database and create tables."""
        cls.db_fd, cls.db_path = tempfile.mkstemp()
        cls.engine = create_engine(
            f'sqlite:///{cls.db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool
        )
        cls.Session = sessionmaker(bind=cls.engine)
        cls.session = cls.Session()
        
        # Create tables
        from app.models.models import Base
        Base.metadata.create_all(cls.engine)
        
        # Create a test document
        cls.doc = Document(
            filename="test.pdf",
            filepath="/path/to/test.pdf",
            title="Test Document"
        )
        cls.session.add(cls.doc)
        cls.session.commit()
        
        # Initialize pipeline
        cls.pipeline = NLPPipeline(cls.session)

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        cls.session.close()
        os.close(cls.db_fd)
        os.unlink(cls.db_path)

    def setUp(self):
        """Set up test case."""
        self.session = self.__class__.session

    def tearDown(self):
        """Clean up after test."""
        self.session.query(Highlight).delete()
        self.session.query(Topic).delete()
        self.session.commit()

    def test_language_detection_english(self):
        """Test language detection for English text."""
        highlight = Highlight(
            document=self.doc,
            text="This is a test highlight in English.",
            page_number=1
        )
        self.session.add(highlight)
        self.session.commit()

        self.pipeline.process_highlights([highlight])
        self.session.refresh(highlight)
        self.assertEqual(highlight.original_language, "en")

    def test_language_detection_german(self):
        """Test language detection and translation for German text."""
        highlight = Highlight(
            document=self.doc,
            text="Dies ist ein Test auf Deutsch.",
            page_number=1
        )
        self.session.add(highlight)
        self.session.commit()

        self.pipeline.process_highlights([highlight])
        self.session.refresh(highlight)
        self.assertEqual(highlight.original_language, "de")
        self.assertIsNotNone(highlight.translated_text)

    def test_topic_modeling(self):
        """Test topic modeling with multiple highlights."""
        # Create test highlights
        highlights = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning models require significant computational resources.",
            "Neural networks are inspired by biological brains.",
            "Data preprocessing is crucial for model performance.",
            "GPUs accelerate deep learning training significantly."
        ]

        for text in highlights:
            highlight = Highlight(document=self.doc, text=text, page_number=1)
            self.session.add(highlight)
        self.session.commit()

        # Update topics
        self.pipeline.process_highlights(self.session.query(Highlight).all())

        # Verify topics were created
        topics = self.session.query(Topic).all()
        self.assertGreater(len(topics), 0)
        
        # Verify highlights have topics assigned
        highlights = self.session.query(Highlight).all()
        for highlight in highlights:
            self.assertIsNotNone(highlight.topic_id)
            self.assertIsNotNone(highlight.topic_confidence)

    def test_semantic_search(self):
        """Test semantic search functionality."""
        # Create test highlights
        highlights = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require training data.",
            "Python is a popular programming language.",
            "Deep learning is revolutionizing AI research.",
            "Natural language processing helps computers understand text."
        ]

        for text in highlights:
            highlight = Highlight(document=self.doc, text=text, page_number=1)
            self.session.add(highlight)
        self.session.commit()

        # Process highlights to generate embeddings
        self.pipeline.process_highlights(self.session.query(Highlight).all())

        # Perform semantic search
        query = "machine learning and artificial intelligence"
        results = self.pipeline.search(query, limit=2)

        self.assertEqual(len(results), 2)
        # Verify results are ordered by relevance
        self.assertIn("machine learning", results[0].text.lower())

if __name__ == '__main__':
    unittest.main() 