"""Natural Language Processing Pipeline for Text Analysis and Topic Modeling.

This module provides functionality for processing and analyzing text data, including:
- Text preprocessing and language detection
- Translation to English when needed
- Topic modeling using BERTopic and semantic search
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pickle
import logging
import unicodedata
import re
import sys

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
from llama_cpp import Llama
import umap
import hdbscan
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from app.models.models import Highlight, Topic, HighlightTopic
from app.models.base import SessionLocal, get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_nlp.log')
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get English stop words and extend with custom words
STOP_WORDS = set(stopwords.words('english'))
CUSTOM_STOP_WORDS = {
    # Personal pronouns
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    
    # Auxiliary verbs
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    
    # Common words that don't add meaning
    'would', 'should', 'could',
    'will', 'shall', 'may', 'might',
    'must', 'can', 'cannot',
    'here', 'there', 'where',
    'when', 'why', 'how',
    'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other',
    'some', 'such', 'than', 'too',
    'very', 'just', 'even', 'also',
    
    # Prepositions and conjunctions
    'about', 'above', 'after', 'again', 'against',
    'before', 'behind', 'below', 'between', 'beyond',
    'but', 'yet', 'however', 'although', 'though',
    
    # Other common words to filter
    'please', 'let', 'know', 'need', 'want',
    'like', 'make', 'way', 'time', 'year',
    'well', 'still', 'back', 'used', 'using',
    'good', 'better', 'best', 'right', 'left',
    'high', 'low', 'many', 'much', 'own',
}

STOP_WORDS.update(CUSTOM_STOP_WORDS)

class NLPPipeline:
    """A pipeline for natural language processing and topic modeling.
    
    This class handles text preprocessing, language detection, translation,
    topic modeling, and semantic search for analyzing text highlights.
    
    Attributes:
        session: Database session for data persistence
        embedding_model: SentenceTransformer model for text embeddings
        topic_model: BERTopic model for topic modeling
        llm: LLaMA model for topic naming and description (optional)
        translator: Translator for non-English text
    """
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the NLP pipeline with necessary components."""
        self.session = SessionLocal()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        # Try to initialize LLaMA (optional)
        self.llm = None
        llama_path = Path(model_dir) / "llama/llama-2-7b-chat.Q4_K_M.gguf"
        logger.info(f"Looking for LLaMA model at: {llama_path}")
        if llama_path.exists():
            try:
                logger.info("Found LLaMA model, attempting to initialize...")
                self.llm = Llama(
                    model_path=str(llama_path),
                    n_gpu_layers=-1,
                    n_ctx=2048,
                    n_batch=512,
                    verbose=True
                )
                logger.info("LLaMA model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LLaMA model: {str(e)}", exc_info=True)
                self.llm = None
        else:
            logger.warning(f"LLaMA model not found at {llama_path}, will use fallback topic naming")
        
        # Initialize topic model with more lenient clustering parameters
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap.UMAP(
                n_neighbors=5,  # Reduced from 15
                n_components=5,
                min_dist=0.0,
                metric='cosine'
            ),
            hdbscan_model=hdbscan.HDBSCAN(
                min_cluster_size=3,  # Reduced from 5
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            ),
            calculate_probabilities=True,
            verbose=True
        )
        
        # Initialize translator
        self.translator = GoogleTranslator(source='auto', target='en')
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Applies comprehensive text preprocessing:
        1. Unicode normalization
        2. Case normalization
        3. Stop words removal
        4. Lemmatization
        5. Custom word filtering
        """
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            lemmatizer.lemmatize(token)  # Lemmatize to base form
            for token in tokens
            if (
                token.isalnum() and  # Only alphanumeric tokens
                len(token) > 2 and  # Skip very short words
                token not in STOP_WORDS and  # Remove stop words
                not token.isnumeric()  # Skip pure numbers
            )
        ]
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def _generate_topic_name(self, keywords: List[Tuple[str, float]]) -> str:
        """Generate a topic name from keywords.
        
        Uses LLaMA if available, otherwise falls back to a simpler approach.
        The generated name follows these rules:
        1. 2-4 words long
        2. Title case format
        3. No articles (a, an, the)
        4. No punctuation except '&' for compound topics
        5. Clear and specific but not too narrow
        
        Args:
            keywords: List of (word, score) tuples
            
        Returns:
            str: Generated topic name
        """
        if self.llm:
            logger.info("Attempting to generate topic name with LLaMA")
            # Use LLaMA for sophisticated topic naming
            keywords_str = ", ".join([word for word, _ in keywords[:5]])
            prompt = f"""You are a topic naming expert. Create a clear, concise name for a topic based on these keywords: {keywords_str}

STRICT RULES - The topic name MUST:
1. Be EXACTLY 2-4 words
2. Use Title Case For Each Word
3. NEVER use articles (a, an, the)
4. ONLY use '&' for compound topics, NO other punctuation
5. Be clear and specific but not too narrow
6. NEVER include meta-words like 'Topic', 'Example', etc.
7. NEVER include any prefixes, suffixes, or explanations
8. NEVER end with phrases like 'Please Let Me Know'

GOOD EXAMPLES:
- Mind & Body
- Personal Growth Strategies
- Scientific Research Methods
- Leadership Skills
- Digital Marketing
- Emotional Intelligence
- Cognitive Development
- Health & Wellness
- Learning & Memory
- Professional Development Skills

BAD EXAMPLES (DO NOT USE):
- The Mind Body Connection (uses article)
- Personal Growth & Development Strategies Today (too long)
- Scientific Research: Methods (uses punctuation)
- A Leadership Framework (uses article)
- Topic: Digital Marketing (includes meta-word)
- Please Let Me Know If This Works (includes meta-phrase)

Your response should be ONLY the topic name, nothing else. No explanations or additional text.
Topic name:"""
            
            try:
                logger.debug(f"Sending prompt to LLaMA: {prompt}")
                response = self.llm(prompt, max_tokens=20)
                raw_response = response["choices"][0]["text"].strip()
                logger.debug(f"Raw LLaMA response: {raw_response}")
                
                # Clean up the response
                response = re.sub(r'^[^a-zA-Z&\s]+|[^a-zA-Z&\s]+$', '', raw_response)  # Remove non-alphanumeric prefixes/suffixes
                response = re.sub(r'\s*&\s*', ' & ', response)  # Standardize & spacing
                response = ' '.join(word.title() for word in response.split())  # Ensure Title Case
                
                # Remove common prefixes and formatting
                prefixes_to_remove = [
                    "Topic name:", "Topic:", "Name:", "Response:",
                    "For example:", "Example:", "E.g.:", "Topic -",
                    "Please provide", "I would suggest", "Based on",
                    "Here's", "How about", "Consider",
                ]
                for prefix in prefixes_to_remove:
                    if response.lower().startswith(prefix.lower()):
                        response = response[len(prefix):].strip()
                
                # Clean up whitespace and limit length
                response = " ".join(response.split())[:50]
                logger.debug(f"Cleaned response: {response}")
                
                # More lenient validation rules
                words = response.split()
                is_valid = (
                    # Allow 2-5 words (increased from 2-4)
                    2 <= len(words) <= 5 and
                    # Allow more flexible capitalization
                    all(word[0].isupper() if word != '&' else True for word in words) and
                    # Still avoid articles and meta-words
                    not any(word.lower() in ['a', 'an', 'the', 'topic', 'example'] for word in words) and
                    # Allow more punctuation but still restrict some
                    not any(char in response for char in '.,;:!?()[]{}/')
                )
                
                if is_valid:
                    logger.info(f"Successfully generated topic name with LLaMA: {response}")
                    return response
                else:
                    logger.warning(f"Generated topic name '{response}' doesn't meet rules, falling back to simple naming")
            except Exception as e:
                logger.error(f"LLaMA topic naming failed: {str(e)}", exc_info=True)
        
        # Simple fallback: Use top 2-3 keywords in Title Case
        logger.info("Using fallback topic naming method")
        top_words = [word.title() for word, _ in keywords[:3]]
        result = " & ".join(top_words[:2])  # Limit to 2 words for fallback
        logger.info(f"Generated fallback topic name: {result}")
        return result
    
    def _generate_topic_keywords(self, topic_words: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Process topic keywords to ensure quality.
        
        Args:
            topic_words: List of (word, score) tuples from BERTopic
            
        Returns:
            List of processed (word, score) tuples
        """
        # Process each keyword
        processed_words = []
        seen_words = set()
        
        for word, score in topic_words:
            # Clean and lemmatize the word
            cleaned_word = self._preprocess_text(word)
            
            # Skip if empty after preprocessing or already seen
            if not cleaned_word or cleaned_word in seen_words:
                continue
            
            # Add to results and mark as seen
            processed_words.append((cleaned_word, score))
            seen_words.add(cleaned_word)
        
        # Sort by score and return top words
        processed_words.sort(key=lambda x: x[1], reverse=True)
        return processed_words[:10]  # Keep top 10 keywords
    
    def process_highlight(self, highlight: Highlight) -> None:
        """Process a single highlight for language, translation, and embedding."""
        try:
            # Clean text
            text = self._preprocess_text(highlight.text)
            
            # Detect language if not already done
            if not highlight.original_language:
                highlight.original_language = self._detect_language(text)
            
            # Translate if not English
            if highlight.original_language != 'en' and not highlight.translated_text:
                highlight.translated_text = self._translate_to_english(
                    text,
                    highlight.original_language
                )
            
            # Use translated text if available, otherwise original
            text_for_embedding = highlight.translated_text or text
            
            # Generate embedding
            embedding = self.embedding_model.encode([text_for_embedding])[0]
            highlight.embedding = pickle.dumps(embedding)
            highlight.has_embedding = True
            
            self.session.add(highlight)
            self.session.commit()
            
        except Exception as e:
            logging.error(f"Error processing highlight: {str(e)}")
            self.session.rollback()
    
    def process_highlights(self, highlights: List[Highlight]) -> None:
        """Process a list of highlights for language detection, translation, and topic modeling.
        
        Args:
            highlights: List of Highlight objects to process
        """
        processed_texts = []
        
        with get_db_session() as session:
            for highlight in highlights:
                try:
                    # Detect language if not already done
                    if not highlight.original_language:
                        highlight.original_language = self._detect_language(highlight.text)
                    
                    # Translate if not in English
                    if highlight.original_language != 'en' and not highlight.translated_text:
                        highlight.translated_text = self._translate_to_english(
                            highlight.text,
                            highlight.original_language
                        )
                    
                    # Use translated text if available, otherwise original
                    text_to_process = highlight.translated_text or highlight.text
                    
                    # Apply preprocessing
                    text_to_process = self._preprocess_text(text_to_process)
                    
                    # Generate embedding if not exists
                    if not highlight.embedding:
                        highlight.embedding = pickle.dumps(
                            self.embedding_model.encode([text_to_process])[0]
                        )
                    
                    processed_texts.append(text_to_process)
                    
                except Exception as e:
                    logging.error(f"Error processing highlight: {str(e)}")
                    continue
            
            session.commit()
            
            # Update topics if we have enough processed texts
            if len(processed_texts) >= 2:
                try:
                    # Prepare embeddings for topic modeling
                    embeddings = []
                    valid_highlights = []
                    for h in highlights:
                        if h.embedding:
                            embeddings.append(pickle.loads(h.embedding))
                            valid_highlights.append(h)
                    
                    if not embeddings:
                        raise ValueError("No valid embeddings generated")

                    # Fit topic model
                    topics, probs = self.topic_model.fit_transform(
                        processed_texts,
                        np.array(embeddings)
                    )
                    
                    # Clear existing topics
                    session.query(Topic).delete()
                    
                    # Create new topics and store their IDs
                    topic_info = self.topic_model.get_topics()
                    topic_id_map = {}  # Map BERTopic IDs to database IDs
                    
                    for topic_id, topic_words in topic_info.items():
                        if topic_id == -1:  # Skip outlier topic
                            continue
                        
                        # Process keywords to ensure quality
                        processed_keywords = self._generate_topic_keywords(topic_words)
                        
                        # Generate topic name using processed keywords
                        topic_name = self._generate_topic_name(processed_keywords)
                        
                        # Create topic
                        topic = Topic(
                            name=topic_name,
                            keywords={word: float(score) for word, score in processed_keywords},
                            description=f"Topic containing keywords: {', '.join([word for word, _ in processed_keywords[:5]])}"
                        )
                        session.add(topic)
                        session.flush()  # Get the ID without committing
                        topic_id_map[topic_id] = topic.id
                    
                    session.commit()
                    
                    # Update highlight topic assignments using the ID mapping
                    topic_ids = list(topic_info.keys())
                    for highlight, _, prob_array in zip(valid_highlights, topics, probs):
                        # Get all topics with probability > 0.05 (lowered from 0.1)
                        # Sort by probability and take top 5 (increased from 3)
                        top_topic_indices = (-prob_array).argsort()[:5]
                        top_topics = [(topic_ids[i], float(prob_array[i])) 
                                    for i in top_topic_indices 
                                    if prob_array[i] > 0.05 and topic_ids[i] != -1]
                        
                        if top_topics:  # If we found any good topics
                            # Clear existing topic assignments for this highlight
                            session.query(HighlightTopic).filter_by(highlight_id=highlight.id).delete()
                            
                            # Normalize confidence scores to make them more balanced
                            total_conf = sum(conf for _, conf in top_topics)
                            if total_conf > 0:
                                normalized_topics = [(tid, conf/total_conf) for tid, conf in top_topics]
                            else:
                                normalized_topics = top_topics
                            
                            # Assign topics with their confidence scores
                            for i, (topic_id, confidence) in enumerate(normalized_topics):
                                if topic_id in topic_id_map:
                                    # Create topic assignment
                                    topic_assignment = HighlightTopic(
                                        highlight_id=highlight.id,
                                        topic_id=topic_id_map[topic_id],
                                        confidence=confidence,
                                        is_primary=(i == 0)  # First topic is primary
                                    )
                                    session.add(topic_assignment)
                                    
                                    # Update primary topic reference in highlights table
                                    if i == 0:
                                        highlight.topic_id = topic_id_map[topic_id]
                                        highlight.topic_confidence = confidence
                    
                    session.commit()
                    logging.info(f"Successfully created {len(topic_info)} topics")
                    
                except Exception as e:
                    logging.error(f"Error in topic modeling: {str(e)}")
                    raise
    
    def search(self, query: str, limit: int = 10) -> List[Highlight]:
        """Search highlights using both semantic similarity and text matching.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of highlights ordered by relevance
        """
        try:
            logging.info(f"Searching for: {query}")
            query_lower = query.lower()
            
            # Generate query embedding for semantic search
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Get all highlights with embeddings
            with get_db_session() as session:
                highlights = session.query(Highlight).filter(Highlight.embedding.isnot(None)).all()
                
                if not highlights:
                    logging.warning("No highlights found with embeddings")
                    return []
                
                # Calculate combined scores using both semantic and text matching
                results = []
                for highlight in highlights:
                    try:
                        # Semantic similarity score (0-1)
                        highlight_embedding = pickle.loads(highlight.embedding)
                        semantic_score = np.dot(query_embedding, highlight_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(highlight_embedding)
                        )
                        
                        # Text matching score (0-1) 
                        text = highlight.translated_text.lower() if highlight.translated_text else highlight.text.lower()
                        text_score = 0.0
                        
                        # Exact phrase match (highest weight)
                        if query_lower in text:
                            text_score = 1.0
                        else:
                            # Word-level matching
                            query_words = set(query_lower.split())
                            text_words = set(text.split())
                            
                            # Exact word matches
                            exact_matches = query_words.intersection(text_words)
                            exact_score = len(exact_matches) / len(query_words) if query_words else 0
                            
                            # Partial word matches (e.g. "test" matches "testing")
                            partial_matches = sum(1 for qw in query_words 
                                               for tw in text_words 
                                               if qw in tw or tw in qw)
                            partial_score = partial_matches / len(query_words) if query_words else 0
                            
                            # Combine exact and partial matches
                            text_score = (0.8 * exact_score) + (0.2 * partial_score)
                        
                        # Final combined score (70% semantic, 30% text)
                        combined_score = (0.7 * semantic_score) + (0.3 * text_score)
                        results.append((highlight, combined_score))
                        
                    except Exception as e:
                        logging.error(f"Error processing highlight {highlight.id}: {str(e)}")
                        continue
                
                # Sort by combined score and return top results
                results.sort(key=lambda x: x[1], reverse=True)
                top_results = [r[0] for r in results[:limit]]
                
                logging.info(f"Found {len(top_results)} results")
                return top_results
                
        except Exception as e:
            logging.error(f"Error in search: {str(e)}")
            raise
    
    def close(self):
        """Close the database session."""
        self.session.close()
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of a text.
        
        Args:
            text: Text to detect language for
            
        Returns:
            str: Language code (e.g., 'en', 'de', etc.)
        """
        try:
            return detect(text)
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            return 'en'  # Default to English on failure
    
    def _translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            str: Translated text
        """
        try:
            if source_lang == 'en':
                return text
            self.translator.source = source_lang
            return self.translator.translate(text)
        except Exception as e:
            logging.warning(f"Translation failed: {e}")
            return text  # Return original text on failure
