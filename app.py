"""
Supple Highlights Web Application

This module serves as the main entry point for the Supple Highlights application, 
a Streamlit-based web interface for managing and analyzing PDF highlights. It provides
functionality for:
- PDF document processing and highlight extraction
- Interactive visualization of highlights and connections
- Topic modeling and text analysis
- Custom styling and user interface components

Dependencies:
    - streamlit: Web application framework
    - PyMuPDF: PDF processing
    - pyvis: Network visualization
    - gensim: Topic modeling
    - nltk: Natural language processing
    - SQLAlchemy: Database operations
"""

import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
from datetime import datetime
from models.base import SessionLocal, engine, Base
from models.models import Document, Highlight
from pyvis.network import Network
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models
import networkx as nx
import numpy as np
from io import BytesIO
import nltk
from nltk.corpus import stopwords
import string
import re
import html
from app.directory_scanner import DirectoryScanner

# Initialize scanner and NLTK components
scanner = DirectoryScanner()

# Download required NLTK data with error handling
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

# Database initialization
Base.metadata.create_all(bind=engine)

# Initialize session state variables for topic modeling
if 'topics' not in st.session_state:
    st.session_state.topics = None
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = None

# Streamlit page configuration
st.set_page_config(
    page_title="Supple Highlights",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Reset Streamlit defaults */
    .stApp > header {
        display: none !important;
    }
    
    /* Main background */
    .stApp {
        background-color: #000000;
    }
    
    /* Header section - no background color */
    .header-container {
        padding: 2rem;
        margin: -4rem -4rem 2rem -4rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #000000;
    }

    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    /* Yellow highlight color */
    .highlight {
        background-color: #FFE600;
        color: #000000;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 500;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
    }

    /* Button text color override */
    .stButton>button p,
    .stButton>button span,
    button[kind="secondary"] p,
    button[kind="secondary"] span {
        color: #000000 !important;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border: 1px solid #FFE600 !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #666666 !important;
    }

    /* Style for text input fields - with higher specificity */
    div[data-testid="stForm"] .stTextInput input,
    section[data-testid="stSidebar"] .stTextInput input,
    .stTextInput input {
        background-color: white !important;
        color: black !important;
    }

    div[data-testid="stForm"] .stTextInput input::placeholder,
    section[data-testid="stSidebar"] .stTextInput input::placeholder,
    .stTextInput input::placeholder {
        color: #666666 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #FFFFFF !important;
    }

    /* Selected tab styling */
    .stTabs [aria-selected="true"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Ensure tab text is black ONLY when selected */
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #000000 !important;
    }

    /* Ensure unselected tabs maintain white text */
    .stTabs [aria-selected="false"] p,
    .stTabs [aria-selected="false"] span {
        color: #FFFFFF !important;
    }

    /* Highlight box styling */
    .highlight-box {
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .highlight-header {
        color: #FFFFFF;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }

    .highlight-header span {
        white-space: nowrap;
    }

    .highlight-content {
        background-color: #1A1A1A;
        border-left: 3px solid #FFE600;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.6;
        color: #FFFFFF;
        white-space: normal;
        word-wrap: break-word;
        font-family: inherit;
        overflow-wrap: break-word;
    }

    .timestamp {
        color: #888888;
        font-size: 0.85rem;
    }

    /* Document header styling */
    .document-header {
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .document-title {
        color: #FFE600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    .document-info {
        color: #FFFFFF;
        font-size: 0.9rem;
        opacity: 0.8;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    /* Make all text white by default */
    .stMarkdown, .stText, p, li {
        color: #FFFFFF !important;
    }

    /* Search section styling */
    .search-title {
        color: #FFE600;
        font-size: 1.1rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Delete button styling */
    button[key*="delete_"] {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.25rem 0.5rem !important;
        border-radius: 4px !important;
    }

    /* Override for any elements with yellow background */
    [style*="background-color: #FFE600"],
    [style*="background-color:#FFE600"] {
        color: #000000 !important;
    }

    /* Ensure all button text is black */
    button:not([key*="delete_"]) {
        color: #000000 !important;
    }

    button:not([key*="delete_"]) * {
        color: #000000 !important;
    }

    /* Specific override for Generate Topics button */
    button:has(p:contains("Generate")) p,
    button:has(p:contains("Generate")) span {
        color: #000000 !important;
    }

    /* Specific override for Knowledge Graph button */
    button:has(p:contains("Knowledge")) p,
    button:has(p:contains("Knowledge")) span {
        color: #000000 !important;
    }

    /* Specific override for Save Directory button */
    button:has(p:contains("Save")) p,
    button:has(p:contains("Save")) span {
        color: #000000 !important;
    }

    /* Specific override for Scan Directory button */
    button:has(p:contains("Scan")) p,
    button:has(p:contains("Scan")) span {
        color: #000000 !important;
    }

    /* Highlight content styling */
    .highlight-content {
        background-color: #1A1A1A;
        border-left: 3px solid #FFE600;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.6;
        color: #FFFFFF;
        white-space: normal;
        word-wrap: break-word;
        font-family: inherit;
        overflow-wrap: break-word;
    }

    /* Ensure highlight boxes don't get treated as code */
    .highlight-box pre,
    .highlight-box code {
        background: none !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        white-space: normal !important;
        font-family: inherit !important;
    }

    /* Override any Streamlit code formatting */
    div[data-testid="stMarkdown"] .highlight-box {
        background: none;
        padding: 0;
        margin: 1rem 0;
    }

    div[data-testid="stMarkdown"] .highlight-content {
        background-color: #1A1A1A;
        white-space: normal;
        font-family: inherit;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_text(text):
    """Preprocess text for topic modeling."""
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = text.split()
        words = [word for word in words if len(word) > 1]
        try:
            stop_words = set(stopwords.words('english'))
            custom_stops = stop_words - {'between', 'through', 'again', 'against', 'once', 'during'}
            words = [word for word in words if word not in custom_stops]
        except Exception as e:
            st.warning(f"Could not load stopwords, continuing without stopword removal: {str(e)}")
        return words if words else []
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return []

def create_topic_model(documents):
    """Create topic model from highlights."""
    try:
        texts = []
        for doc in documents:
            # Combine highlights but maintain some structure with periods
            doc_text = ". ".join([h.text for h in doc.highlights])
            words = preprocess_text(doc_text)
            if words:
                texts.append(words)
        
        if not texts:
            st.error("No valid text found to process for topic modeling.")
            return None
        
        dictionary = corpora.Dictionary(texts)
        # Adjust filter parameters to be more strict
        dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=20000)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        if not corpus or not any(corp for corp in corpus):
            st.error("No valid terms found after preprocessing.")
            return None
        
        num_topics = min(15, len(documents))
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=50,  # Increased passes for better topic separation
            random_state=42,
            minimum_probability=0.01,  # Set minimum probability threshold
            alpha='asymmetric',  # Changed to asymmetric for better topic distribution
            eta='auto',
            chunksize=2000,
            iterations=1000  # Increased iterations for better convergence
        )
        
        st.session_state.topics = lda_model.show_topics(formatted=False, num_words=15)
        st.session_state.topic_model = lda_model
        st.session_state.dictionary = dictionary
        
        st.success(f"Generated {num_topics} topics successfully!")
        return lda_model
        
    except Exception as e:
        st.error(f"Error in topic modeling: {str(e)}")
        return None

def create_knowledge_graph(documents):
    """Create a knowledge graph from documents."""
    try:
        G = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")
        G.set_options("""
        var options = {
            "nodes": {
                "font": {"size": 20, "color": "white"},
                "scaling": {"min": 20, "max": 60}
            },
            "edges": {
                "color": {"color": "#FFE600", "inherit": false},
                "smooth": {"type": "continuous"},
                "width": 0.5
            },
            "physics": {
                "barnesHut": {"gravitationalConstant": -80000, "springLength": 250},
                "stabilization": {"iterations": 25}
            }
        }
        """)
        
        documents_with_highlights = [doc for doc in documents if doc.highlights]
        
        if not documents_with_highlights:
            st.warning("No documents with highlights found to create the knowledge graph.")
            return None
        
        # Calculate maximum number of highlights for scaling
        max_highlights = max(len(doc.highlights) for doc in documents_with_highlights)
        
        # Add document nodes with scaled sizes
        for doc in documents_with_highlights:
            num_highlights = len(doc.highlights)
            # Scale node size based on number of highlights
            size = 20 + (num_highlights / max_highlights) * 40
            G.add_node(
                f"doc_{doc.id}",
                label=doc.filename,
                title=f"Document: {doc.filename}\nHighlights: {num_highlights}",
                color="#FFE600",
                size=size
            )
        
        if st.session_state.topics:
            # First, calculate topic strengths for all documents
            doc_topic_weights = {}
            for doc in documents_with_highlights:
                doc_text = ". ".join([h.text for h in doc.highlights])
                tokens = preprocess_text(doc_text)
                bow = st.session_state.dictionary.doc2bow(tokens)
                # Get topic distribution
                doc_topics = st.session_state.topic_model.get_document_topics(
                    bow, 
                    minimum_probability=0.01
                )
                doc_topic_weights[doc.id] = doc_topics
            
            # Calculate threshold based on document length
            def calculate_threshold(num_highlights):
                # Dynamic threshold that decreases as number of highlights increases
                base_threshold = 0.1  # 10% base threshold
                return max(0.05, base_threshold / (1 + num_highlights/10))
            
            # Add topic nodes and edges
            for idx, (topic_id, word_scores) in enumerate(st.session_state.topics):
                has_connections = False
                edges_to_add = []
                
                for doc in documents_with_highlights:
                    doc_topics = doc_topic_weights[doc.id]
                    num_highlights = len(doc.highlights)
                    threshold = calculate_threshold(num_highlights)
                    
                    for t_id, weight in doc_topics:
                        if t_id == idx and weight > threshold:
                            has_connections = True
                            edges_to_add.append((doc.id, weight))
                
                if has_connections:
                    # Get top words and their scores for better topic representation
                    top_words = ", ".join([f"{word} ({score:.3f})" for word, score in word_scores[:3]])
                    G.add_node(
                        f"topic_{idx}",
                        label=f"Topic {idx+1}",
                        title=f"Topic {idx+1}\n{top_words}",
                        color="#FFFFFF",
                        size=25
                    )
                    
                    for doc_id, weight in edges_to_add:
                        # Scale edge width based on weight
                        edge_width = weight * 5  # Increase edge width for visibility
                        G.add_edge(
                            f"doc_{doc_id}",
                            f"topic_{idx}",
                            value=edge_width,
                            title=f"Strength: {weight:.3f}",
                            width=edge_width
                        )
        
        return G
    except Exception as e:
        st.error(f"Error creating knowledge graph: {str(e)}")
        return None

def delete_document(doc_id):
    """Delete a document and its highlights from the database."""
    try:
        db = SessionLocal()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            st.session_state.topics = None
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False
    finally:
        db.close()

def view_saved_highlights():
    """View highlights saved in the database with advanced features."""
    db = SessionLocal()
    try:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        
        if not documents:
            st.info("No highlights saved yet. Configure a directory and scan for PDFs to get started!")
            return
        
        tab1, tab2, tab3 = st.tabs(["📚 Highlights", "📊 Topics", "🕸️ Graph"])
        
        with tab1:
            st.markdown("""
            <div style="background-color: #2D2D2D; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
                <h3 class="search-title">🔍 Search Highlights</h3>
            </div>
            """, unsafe_allow_html=True)
            
            search_query = st.text_input("", 
                                    placeholder="Type to search across all highlights...",
                                    key="search",
                                    label_visibility="collapsed")
            
            for doc in documents:
                highlights_to_show = doc.highlights
                if search_query:
                    highlights_to_show = [h for h in doc.highlights if search_query.lower() in h.text.lower()]
                
                if highlights_to_show:
                    toggle_key = f"toggle_{doc.id}"
                    if toggle_key not in st.session_state:
                        st.session_state[toggle_key] = True
                    
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        if st.button("", key=f"toggle_btn_{doc.id}"):
                            st.session_state[toggle_key] = not st.session_state[toggle_key]
                            st.rerun()
                        
                        st.markdown(f"""
                        <div class="document-header">
                            <div class="document-title">📄 {html.escape(doc.filename)}</div>
                            <div class="document-info">
                                <span>{len(highlights_to_show)} highlights</span>
                                <span>|</span>
                                <span class="timestamp">Created: {doc.created_at.strftime('%Y-%m-%d %H:%M:%S')}</span>
                                <span>|</span>
                                <span class="timestamp">Last updated: {doc.updated_at.strftime('%Y-%m-%d %H:%M:%S')}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{doc.id}"):
                            if delete_document(doc.id):
                                st.success(f"Deleted {doc.filename}")
                                st.rerun()
                    
                    if st.session_state[toggle_key]:
                        for highlight in highlights_to_show:
                            # Display highlight and metadata in a single container
                            st.markdown(
                                f"""<div style="background-color: #2D2D2D; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem;">
                                    <div style="color: #666666; font-size: 0.8rem; margin-bottom: 0.5rem;">
                                        Page {highlight.page_number} | Added: {highlight.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                                    </div>
                                    <div style="color: #FFFFFF;">
                                        {highlight.text.strip()}
                                    </div>
                                </div>""",
                                unsafe_allow_html=True
                            )
        
        with tab2:
            if st.button("Generate Topics") or st.session_state.topics is None:
                with st.spinner("Generating topics..."):
                    create_topic_model(documents)
            
            if st.session_state.topics:
                for idx, (_, word_scores) in enumerate(st.session_state.topics):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"### Topic {idx + 1}")
                        words = dict(word_scores)
                        for word, score in list(words.items())[:5]:
                            st.write(f"- {word}: {score:.3f}")
                    
                    with col2:
                        try:
                            wordcloud = WordCloud(
                                width=400,
                                height=200,
                                background_color='#111111',
                                colormap='YlOrRd'
                            ).generate_from_frequencies(words)
                            
                            plt.figure(figsize=(8, 4), facecolor='#111111')
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(plt)
                            plt.close()
                        except Exception as e:
                            st.error(f"Error creating word cloud: {str(e)}")
        
        with tab3:
            if st.button("Generate Knowledge Graph") or 'graph' not in st.session_state:
                with st.spinner("Creating knowledge graph..."):
                    graph = create_knowledge_graph(documents)
                    if graph:
                        try:
                            graph.save_graph("temp_graph.html")
                            with open("temp_graph.html", "r", encoding="utf-8") as f:
                                graph_html = f.read()
                            st.components.v1.html(graph_html, height=700)
                            os.remove("temp_graph.html")
                        except Exception as e:
                            st.error(f"Error displaying graph: {str(e)}")
    finally:
        db.close()

# Main app layout
st.title("Supple Highlights")

# Add custom CSS for text input fields
st.markdown("""
<style>
    /* Style for text input fields */
    .stTextInput input {
        color: black !important;
        background-color: white !important;
    }
    .stTextInput input::placeholder {
        color: #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

# Directory Configuration Section in Sidebar
st.sidebar.header("Directory Configuration")

watched_dir = scanner.get_watched_directory()
current_dir = watched_dir.path if watched_dir else None

if current_dir:
    st.sidebar.text("Currently watching:")
    st.sidebar.code(current_dir)
    if watched_dir.last_scan:
        st.sidebar.text(f"Last scan: {watched_dir.last_scan.strftime('%Y-%m-%d %H:%M:%S')}")

# Directory selection
new_dir = st.sidebar.text_input("PDF Directory Path", value=current_dir or "")
if st.sidebar.button("Save Directory"):
    try:
        scanner.set_watched_directory(new_dir)
        st.sidebar.success("Directory updated successfully!")
    except ValueError as e:
        st.sidebar.error(str(e))

# Scan button
if st.sidebar.button("Scan Directory"):
    if not watched_dir:
        st.sidebar.error("Please configure a directory first")
    else:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        def update_progress(filename, count):
            status_text.text(f"Processing: {filename}")
            progress_bar.progress(min(1.0, count / max(1, len([f for f in os.listdir(watched_dir.path) if f.lower().endswith('.pdf')]))))
        
        try:
            processed_docs = scanner.scan_directory(callback=update_progress)
            progress_bar.progress(1.0)
            if processed_docs:
                st.sidebar.success(f"Processed {len(processed_docs)} new PDF files")
            else:
                st.sidebar.info("No new PDF files to process")
        except Exception as e:
            st.sidebar.error(f"Error during scan: {str(e)}")
        finally:
            status_text.empty()
            progress_bar.empty()

# Display the main content
view_saved_highlights() 