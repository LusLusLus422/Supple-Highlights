"""Natural Language Processing Pipeline for Text Analysis and Topic Modeling.

This module provides functionality for processing and analyzing text data, including:
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Topic modeling using LDA (Latent Dirichlet Allocation)
- Database integration for storing and retrieving text highlights and topics

The module uses NLTK for text processing and Gensim for topic modeling.
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import numpy as np
from typing import List, Tuple, Dict
from models.models import Highlight, Topic
from models.base import Session

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class NLPPipeline:
    """A pipeline for natural language processing and topic modeling.
    
    This class handles text preprocessing, topic modeling, and database operations
    for analyzing and categorizing text highlights.
    
    Attributes:
        session: Database session for data persistence
        lemmatizer: WordNet lemmatizer instance
        stop_words: Set of English stop words
        vectorizer: TF-IDF vectorizer for text feature extraction
    """

    def __init__(self):
        """Initialize the NLP pipeline with necessary components."""
        self.session = Session()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text data for analysis.
        
        Performs tokenization, stopword removal, and lemmatization on input text.
        
        Args:
            text: Raw input text to process.
            
        Returns:
            List of processed tokens.
            
        Example:
            >>> pipeline.preprocess_text("The quick brown fox jumps")
            ['quick', 'brown', 'fox', 'jump']
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        
        return tokens
    
    def create_topic_model(self, num_topics: int = 5) -> None:
        """Create and apply topic model to existing highlights.
        
        Builds an LDA topic model from all highlights in the database and
        assigns the most probable topic to each highlight.
        
        Args:
            num_topics: Number of topics to generate in the model. Defaults to 5.
            
        Raises:
            SQLAlchemyError: If database operations fail
            ValueError: If no highlights exist in database
        """
        # Get all highlights
        highlights = self.session.query(Highlight).all()
        texts = [highlight.text for highlight in highlights]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create dictionary and corpus for LDA
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model with specified parameters
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=15  # Number of passes through corpus during training
        )
        
        # Create topics in database
        topics = []
        for topic_id in range(num_topics):
            topic_words = dict(lda_model.show_topic(topic_id))
            topic_name = ", ".join(list(topic_words.keys())[:5])
            topic = Topic(
                name=f"Topic {topic_id + 1}",
                description=topic_name
            )
            topics.append(topic)
        
        self.session.add_all(topics)
        self.session.commit()
        
        # Assign most probable topic to each highlight
        for highlight, bow in zip(highlights, corpus):
            topic_dist = lda_model.get_document_topics(bow)
            if topic_dist:
                main_topic_id = max(topic_dist, key=lambda x: x[1])[0]
                highlight.topic = topics[main_topic_id]
        
        self.session.commit()
    
    def close(self):
        """Close the database session.
        
        Should be called when the pipeline is no longer needed to free
        database resources.
        """
        self.session.close() 