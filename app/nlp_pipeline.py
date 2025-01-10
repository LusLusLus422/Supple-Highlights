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
    def __init__(self):
        self.session = Session()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing."""
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
        """Create topic model from highlights."""
        # Get all highlights
        highlights = self.session.query(Highlight).all()
        texts = [highlight.text for highlight in highlights]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=15
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
        
        # Assign topics to highlights
        for highlight, bow in zip(highlights, corpus):
            topic_dist = lda_model.get_document_topics(bow)
            if topic_dist:
                # Assign the most probable topic
                main_topic_id = max(topic_dist, key=lambda x: x[1])[0]
                highlight.topic = topics[main_topic_id]
        
        self.session.commit()
    
    def close(self):
        """Close the database session."""
        self.session.close() 