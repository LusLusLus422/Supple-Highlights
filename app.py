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
    - nltk: Natural language processing
    - SQLAlchemy: Database operations
"""

import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
from datetime import datetime
from app.models.base import SessionLocal, get_db_session, engine, Base
from app.models.models import Document, Highlight, Topic
from pyvis.network import Network
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from io import BytesIO
import nltk
from nltk.corpus import stopwords
import string
import re
import html
from app.directory_scanner import DirectoryScanner
from app.nlp_pipeline import NLPPipeline
import logging
from contextlib import contextmanager

# Initialize scanner, NLP pipeline and NLTK components
scanner = DirectoryScanner()
nlp_pipeline = NLPPipeline()

# Database session management
@contextmanager
def get_db():
    """Database session context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create global session state for database session
if 'db_session' not in st.session_state:
    st.session_state.db_session = SessionLocal()

# Use session from session state
session = st.session_state.db_session

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

def create_topic_model(highlights):
    """Create a topic model from the given highlights."""
    try:
        if len(highlights) < 2:
            st.warning("At least 2 highlights are needed for topic modeling.")
            return

        # Process highlights using NLP pipeline
        with st.spinner("Processing highlights..."):
            nlp_pipeline.process_highlights(highlights)
            st.success("Topics generated successfully!")
            
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        st.error(f"Error accessing database: {str(e)}")
        return False
    
    return True

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
        
        # Add topic nodes and edges
        topics_added = set()
        for doc in documents_with_highlights:
            for highlight in doc.highlights:
                if highlight.topic and highlight.topic.name not in topics_added:
                    # Add topic node
                    topic_name = highlight.topic.name
                    G.add_node(
                        f"topic_{topic_name}",
                        label=topic_name,
                        title=f"Topic: {topic_name}\nKeywords: {', '.join(highlight.topic.keywords.keys())}",
                        color="#FFFFFF",
                        size=25
                    )
                    topics_added.add(topic_name)
                
                if highlight.topic:
                    # Add edge with confidence score
                    G.add_edge(
                        f"doc_{doc.id}",
                        f"topic_{highlight.topic.name}",
                        value=highlight.topic_confidence * 5,
                        title=f"Confidence: {highlight.topic_confidence:.3f}",
                        width=highlight.topic_confidence * 5
                    )
        
        return G
    except Exception as e:
        st.error(f"Error creating knowledge graph: {str(e)}")
        return None

def delete_document(doc_id):
    """Delete a document and its highlights from the database."""
    try:
        with get_db_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                session.delete(doc)
                st.session_state.topics = None
                return True
        return False
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        st.error("An error occurred while deleting the document. Please try again.")
        return False

def view_saved_highlights():
    """View highlights saved in the database with advanced features."""
    try:
        with get_db_session() as session:
            # Only get documents that have highlights
            documents = session.query(Document).filter(Document.highlights.any()).order_by(Document.created_at.desc()).all()
            
            if not documents:
                st.info("No highlights saved yet. Configure a directory and scan for PDFs to get started!")
                return
            
            tab1, tab2, tab3 = st.tabs(["📚 Highlights", "📊 Topics", "🕸️ Graph"])
            
            with tab1:
                st.markdown("""
                <div style="background-color: #2D2D2D; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
                    <h3 class="search-title">📚 Documents & Highlights</h3>
                </div>
                """, unsafe_allow_html=True)
                
                search_query = st.text_input("", 
                                        placeholder="Type to search across all highlights...",
                                        key="search",
                                        label_visibility="collapsed")
                
                if search_query:
                    # Use semantic search from NLP pipeline
                    results = nlp_pipeline.search(search_query)
                    highlights_to_show = results
                    
                    # Group search results by document
                    highlights_by_doc = {}
                    for highlight in highlights_to_show:
                        doc = highlight.document
                        if doc not in highlights_by_doc:
                            highlights_by_doc[doc] = []
                        highlights_by_doc[doc].append(highlight)
                        
                    # Display search results grouped by document
                    for doc, highlights in highlights_by_doc.items():
                        with st.expander(f"📄 {doc.filename} ({len(highlights)} matches)", expanded=True):
                            for highlight in highlights:
                                st.markdown(f"""
                                <div style="background-color: #2D2D2D; padding: 1rem; border-radius: 4px; margin: 0.5rem 0;">
                                    <div style="color: #FFFFFF;">
                                        {highlight.text.strip()}
                                    </div>
                                    <div style="color: #999999; font-size: 0.8em; margin-top: 0.5rem;">
                                        Page {highlight.page_number} | Created: {highlight.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    # Show all documents with their highlights
                    for doc in documents:
                        with st.expander(f"📄 {doc.filename} ({len(doc.highlights)} highlights)", expanded=False):
                            for highlight in doc.highlights:
                                st.markdown(f"""
                                <div style="background-color: #2D2D2D; padding: 1rem; border-radius: 4px; margin: 0.5rem 0;">
                                    <div style="color: #FFFFFF;">
                                        {highlight.text.strip()}
                                    </div>
                                    <div style="color: #999999; font-size: 0.8em; margin-top: 0.5rem;">
                                        Page {highlight.page_number} | Created: {highlight.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("""
                <div style="background-color: #2D2D2D; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
                    <h3 class="topics-title">📊 Topic Modeling</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Get all highlights for topic modeling
                all_highlights = []
                for doc in documents:
                    all_highlights.extend(doc.highlights)
                
                if len(all_highlights) < 2:
                    st.warning("Add at least 2 highlights to enable topic modeling.")
                else:
                    if st.button("Generate Topics"):
                        create_topic_model(all_highlights)
                
                # Display existing topics
                with get_db_session() as session:
                    topics = session.query(Topic).all()
                    if topics:
                        for topic in topics:
                            with st.expander(f"📌 Topic: {topic.name}", expanded=False):
                                st.write("**Keywords:**", ", ".join(topic.keywords.keys()))
                                st.write("**Description:**", topic.description)
                                st.markdown("#### Related Highlights:")
                                
                                # Display related highlights
                                if topic.highlights:
                                    for highlight in topic.highlights:
                                        st.markdown(f"""
                                        <div style="background-color: #2D2D2D; padding: 1rem; border-radius: 4px; margin: 0.5rem 0;">
                                            <div style="color: #FFFFFF;">
                                                {highlight.text.strip()}
                                            </div>
                                            <div style="color: #999999; font-size: 0.8em; margin-top: 0.5rem;">
                                                From: {highlight.document.filename} (Page {highlight.page_number})
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No highlights associated with this topic.")
                            
                            # Add a separator between topics
                            st.markdown("---")
            
            with tab3:
                st.markdown("""
                <div style="background-color: #2D2D2D; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
                    <h3 class="graph-title">🕸️ Knowledge Graph</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Generate Knowledge Graph"):
                    with st.spinner("Creating knowledge graph..."):
                        G = create_knowledge_graph(documents)
                        if G:
                            # Save graph to HTML file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                                G.save_graph(f.name)
                                # Display the graph in an iframe
                                st.components.v1.html(open(f.name, 'r').read(), height=600)
                
    except Exception as e:
        logging.error(f"Error viewing saved highlights: {str(e)}")
        st.error("An error occurred while viewing highlights. Please try again.")

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