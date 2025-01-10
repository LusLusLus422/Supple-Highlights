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

# Custom styling
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
    
    /* Radio buttons */
    .stRadio > div {
        background-color: transparent !important;
    }
    
    .stRadio > div > div > label {
        color: #FFFFFF !important;
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
        background-color: #FFE600;
        color: #000000;
        border: none;
        font-weight: bold;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: transparent;
        color: #FFFFFF;
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background-color: transparent !important;
        color: #FFFFFF !important;
        border: 1px solid #FFE600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: transparent;
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
    }
    
    .highlight-content {
        background-color: #1A1A1A;
        border-left: 3px solid #FFE600;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.5;
        color: #FFFFFF;
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
    }
    
    /* Text input placeholder */
    .stTextInput > div > div > input::placeholder {
        color: #FFFFFF !important;
        opacity: 0.6 !important;
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
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #3D3D3D !important;
        border-radius: 4px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #FFE600 !important;
        box-shadow: 0 0 0 1px #FFE600 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton > button:hover {
        background-color: #FFD700 !important;
        color: #000000 !important;
        border: none !important;
    }
    
    /* Tab styling */
    .stTabs [aria-selected="true"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    /* Button styling */
    button[kind="primary"], button[kind="secondary"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    /* Specific button overrides */
    .stButton button, .stButton > button {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    /* Generate Topics button */
    button[data-testid="baseButton-secondary"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    /* Save Highlights button */
    button[data-testid="baseButton-primary"] {
        background-color: #FFE600 !important;
        color: #000000 !important;
        font-weight: bold !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1.1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Ensure all yellow background elements have black text */
    [style*="background-color: #FFE600"],
    [style*="background-color:#FFE600"] {
        color: #000000 !important;
    }
    
    /* Button styling - more specific selectors */
    .stButton > button p,
    .stButton button p,
    button[data-testid="baseButton-secondary"] p,
    button[kind="secondary"] p {
        color: #000000 !important;
    }
    
    .stButton > button span,
    .stButton button span,
    button[data-testid="baseButton-secondary"] span,
    button[kind="secondary"] span {
        color: #000000 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span {
        color: #000000 !important;
    }
    
    /* Override any white text on yellow background */
    [style*="background-color: #FFE600"] *,
    [style*="background-color:#FFE600"] *,
    [style*="background-color: rgb(255, 230, 0)"] *,
    [style*="background-color:rgb(255, 230, 0)"] * {
        color: #000000 !important;
    }
    
    /* Specific override for Generate Topics button */
    button:has(p:contains("Generate")) p,
    button:has(p:contains("Generate")) span {
        color: #000000 !important;
    }
    
    /* Specific override for Generate Knowledge Graph button */
    button:has(p:contains("Knowledge")) p,
    button:has(p:contains("Knowledge")) span {
        color: #000000 !important;
    }
    
    /* Toggle switch container */
    .toggle-container {
        position: relative;
        margin-bottom: 1rem;
    }
    
    /* Toggle switch styling */
    .toggle-switch {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
        cursor: pointer;
    }
    
    /* Remove the invisible button styling */
    .stButton > button {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        opacity: 1 !important;
        height: auto !important;
        width: auto !important;
        position: relative !important;
        z-index: 1 !important;
        margin-bottom: 0 !important;
    }
    
    /* Document info container */
    .document-info-container {
        flex-grow: 1;
    }
    
    /* Toggle icon */
    .toggle-icon {
        color: #FFE600;
        font-size: 1.5rem;
        margin-left: 1rem;
        width: 24px;
        text-align: center;
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
    
    /* Generate Topics button styling */
    button:has(p:contains("Generate Topics")) {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        margin: 1rem 0 !important;
        width: auto !important;
    }
    
    /* Knowledge Graph button styling */
    button:has(p:contains("Knowledge Graph")) {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        margin: 1rem 0 !important;
        width: auto !important;
    }
    
    /* Save Highlights button styling */
    button:has(p:contains("Save Highlights")) {
        background-color: #FFE600 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.75rem 1.5rem !important;
        margin: 1rem 0 !important;
        width: auto !important;
        font-size: 1.1rem !important;
    }
    
    /* Toggle button styling */
    button[key*="toggle_btn_"] {
        background-color: transparent !important;
        width: 100% !important;
        height: 100% !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        opacity: 0 !important;
        cursor: pointer !important;
    }
    
    /* Container styling */
    div[data-testid="stButton"] {
        width: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# App header with logo
st.markdown("""
<div class="header-container">
    <h1 style="margin: 0;">
        <span style="font-size: 2.5rem;">© Supple</span>
        <span class="highlight" style="font-size: 2.5rem;">Highlights</span>
    </h1>
    <p style="font-size: 1.2rem; color: #FFFFFF; margin-top: 0.5rem;">
        Explorative Personalized Knowledge Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# Download required NLTK data
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize session state
if 'topics' not in st.session_state:
    st.session_state.topics = None
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = None

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def extract_highlights(pdf_path):
    """Extract highlights from a PDF file."""
    highlights = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        annots = page.annots()
        if annots:
            for annot in annots:
                if annot.type[0] == 8:  # Highlight annotation
                    rect = annot.rect
                    highlight_text = page.get_text("text", clip=rect)
                    highlights.append({
                        'page': page_num + 1,
                        'text': highlight_text.strip(),
                        'rect': rect
                    })
    
    doc.close()
    return highlights

def save_highlights(filename, highlights):
    """Save document and highlights to database."""
    db = get_db()
    
    # Create document record
    doc = Document(
        filename=filename,
        title=filename,
        created_at=datetime.utcnow()
    )
    db.add(doc)
    db.flush()
    
    # Create highlight records
    for highlight in highlights:
        h = Highlight(
            document_id=doc.id,
            text=highlight['text'],
            page_number=highlight['page'],
            rect_x0=highlight['rect'].x0,
            rect_y0=highlight['rect'].y0,
            rect_x1=highlight['rect'].x1,
            rect_y1=highlight['rect'].y1
        )
        db.add(h)
    
    db.commit()
    return doc.id

def preprocess_text(text):
    """Preprocess text for topic modeling."""
    try:
        # Basic text cleaning
        text = text.lower()
        
        # Remove special characters but keep letters and numbers
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Split into words
        words = text.split()
        
        # Remove very short words (length < 2)
        words = [word for word in words if len(word) > 1]
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            # Remove common stopwords but keep potentially important ones
            custom_stops = stop_words - {'between', 'through', 'again', 'against', 'once', 'during'}
            words = [word for word in words if word not in custom_stops]
        except Exception as e:
            st.warning(f"Could not load stopwords, continuing without stopword removal: {str(e)}")
        
        if not words:
            st.warning("No valid words found after preprocessing. The text might be too short.")
            return []
            
        return words
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return []

def create_topic_model(documents):
    """Create topic model from highlights."""
    try:
        # Prepare text data
        texts = []
        for doc in documents:
            # Combine all highlights
            doc_text = " ".join([h.text for h in doc.highlights])
            # Preprocess text
            words = preprocess_text(doc_text)
            if words:  # Only add if we got valid words
                texts.append(words)
        
        if not texts:
            st.error("No valid text found to process for topic modeling. Please ensure your highlights contain meaningful text.")
            return None
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(texts)
        
        # Less strict filtering of terms
        dictionary.filter_extremes(no_below=1, no_above=0.99, keep_n=20000)
        
        # Create corpus
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        if not corpus or not any(corp for corp in corpus):
            st.error("No valid terms found after preprocessing. Please ensure your highlights contain meaningful text.")
            return None
        
        # Train LDA model with adjusted parameters
        num_topics = min(5, len(documents))  # Allow more topics
        lda_model = models.LdaModel(
            corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=30,  # Increase passes for better convergence
            random_state=42,
            minimum_probability=0.0,
            update_every=1,
            chunksize=10,
            alpha='symmetric',  # Use symmetric alpha for more balanced topics
            eta='auto'  # Automatically learn the eta parameter
        )
        
        # Save to session state
        st.session_state.topics = lda_model.show_topics(formatted=False, num_words=15)  # Show more words per topic
        st.session_state.topic_model = lda_model
        st.session_state.dictionary = dictionary
        
        # Show success message
        st.success("Topics generated successfully!")
        return lda_model
        
    except Exception as e:
        st.error(f"Error in topic modeling: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def create_knowledge_graph(documents):
    """Create a knowledge graph from documents."""
    try:
        # Create network
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
        
        # Add document nodes
        for doc in documents:
            G.add_node(
                f"doc_{doc.id}",
                label=doc.filename,
                title=f"Document: {doc.filename}\nHighlights: {len(doc.highlights)}",
                color="#FFE600",
                size=30
            )
        
        # Add topic nodes if topics exist
        if st.session_state.topics:
            for idx, (topic_id, word_scores) in enumerate(st.session_state.topics):
                # Get top words for topic label
                top_words = ", ".join([word for word, _ in word_scores[:3]])
                G.add_node(
                    f"topic_{idx}",
                    label=f"Topic {idx+1}",
                    title=f"Top words: {top_words}",
                    color="#FFFFFF",
                    size=25
                )
                
                # Connect topics to documents
                for doc in documents:
                    doc_text = " ".join([h.text for h in doc.highlights])
                    tokens = preprocess_text(doc_text)
                    bow = st.session_state.dictionary.doc2bow(tokens)
                    doc_topics = st.session_state.topic_model.get_document_topics(bow)
                    
                    for topic_id, weight in doc_topics:
                        if topic_id == idx and weight > 0.05:  # Lowered threshold from 0.1 to 0.05
                            G.add_edge(
                                f"doc_{doc.id}",
                                f"topic_{idx}",
                                value=weight * 10,
                                title=f"Strength: {weight:.2f}"
                            )
        
        return G
    except Exception as e:
        st.error(f"Error creating knowledge graph: {str(e)}")
        return None

        return False

def delete_document(doc_id):
    """Delete a document and its highlights from the database."""
    try:
        db = get_db()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            db.delete(doc)  # This will also delete associated highlights due to cascade
            db.commit()
            # Reset topic model since we removed a document
            st.session_state.topics = None
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def view_saved_highlights():
    """View highlights saved in the database with advanced features."""
    db = get_db()
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    
    if not documents:
        st.info("No highlights saved yet. Upload a PDF to get started!")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📚 Highlights", "📊 Topics", "🕸️ Graph"])
    
    with tab1:
        # Search and filter section with improved styling
        st.markdown("""
        <div style="background-color: #2D2D2D; padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem; border: 1px solid #3D3D3D;">
            <h3 class="search-title">🔍 Search Highlights</h3>
        </div>
        """, unsafe_allow_html=True)
        search_query = st.text_input("", 
                                   placeholder="Type to search across all highlights...",
                                   key="search",
                                   label_visibility="collapsed")
        
        # Display filtered highlights
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
                    st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
                    
                    # Create the toggle button first
                    if st.button("", key=f"toggle_btn_{doc.id}", help="Toggle highlights"):
                        st.session_state[toggle_key] = not st.session_state[toggle_key]
                        st.rerun()
                    
                    # Display the header with toggle state
                    st.markdown(f"""
                    <div class="toggle-switch">
                        <div class="document-info-container">
                            <div class="document-title">
                                <span>📄 {doc.filename}</span>
                                <span style="color: #FFE600;">({len(highlights_to_show)} highlights)</span>
                            </div>
                            <div style="color: #888888; font-size: 0.9rem;">
                                Added on: {doc.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                            </div>
                        </div>
                        <div class="toggle-icon">
                            {'▼' if st.session_state[toggle_key] else '▶'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    if st.button("🗑️", key=f"delete_{doc.id}"):
                        if delete_document(doc.id):
                            st.success(f"Deleted {doc.filename}")
                            st.rerun()
                        continue
                
                # Display highlights if expanded
                if st.session_state[toggle_key]:
                    st.markdown('<div class="highlights-container">', unsafe_allow_html=True)
                    for highlight in highlights_to_show:
                        st.markdown(f"""
                        <div class="highlight-box">
                            <div class="highlight-header">
                                Page {highlight.page_number}
                            </div>
                            <div class="highlight-content">
                                {highlight.text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Create or update topic model
        if st.button("Generate Topics") or st.session_state.topics is None:
            with st.spinner("Generating topics..."):
                create_topic_model(documents)
        
        # Display topics and word clouds
        if st.session_state.topics:
            for idx, (_, word_scores) in enumerate(st.session_state.topics):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <h3 style="color: #FFE600;">Topic {idx + 1}</h3>
                    """, unsafe_allow_html=True)
                    words = dict(word_scores)
                    for word, score in list(words.items())[:5]:
                        st.write(f"- {word}: {score:.3f}")
                
                with col2:
                    try:
                        wordcloud = WordCloud(
                            width=400,
                            height=200,
                            background_color='#111111',
                            colormap='YlOrRd',
                            color_func=lambda *args, **kwargs: "#FFE600"
                        ).generate_from_frequencies(words)
                        
                        plt.figure(figsize=(8, 4), facecolor='#111111')
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(plt)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error creating word cloud: {str(e)}")
    
    with tab3:
        # Knowledge Graph
        if st.button("Generate Knowledge Graph", type="primary") or 'graph' not in st.session_state:
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
                else:
                    st.error("Failed to create knowledge graph. Please try again.")

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: #FFFFFF; margin: 0;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Add the radio buttons
    page = st.radio(
        "Navigation",
        ["Upload PDF", "View Saved Highlights"],
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div style="margin-top: 2rem;">
        <h2 style="color: #FFFFFF;">How to use</h2>
        <ol style="color: #FFFFFF;">
            <li>Prepare a PDF with highlighted text</li>
            <li>Click 'Browse files' to upload your PDF</li>
            <li>View extracted highlights</li>
            <li>Save highlights to database</li>
        </ol>
    </div>
    
    <div style="margin-top: 1rem;">
        <h2 style="color: #FFFFFF;">Features</h2>
        <ul style="list-style-type: none; padding-left: 0; color: #FFFFFF;">
            <li style="margin-bottom: 0.5rem;">📝 <span class="highlight">Extract and save highlights</span></li>
            <li style="margin-bottom: 0.5rem;">🔍 <span style="color: #FFE600;">Search across all highlights</span></li>
            <li style="margin-bottom: 0.5rem;">📊 <span style="color: #FFE600;">Topic modeling and visualization</span></li>
            <li style="margin-bottom: 0.5rem;">🕸️ <span style="color: #FFE600;">Interactive knowledge graph</span></li>
        </ul>
    </div>
    
    <div style="margin-top: 1rem;">
        <h2 style="color: #FFFFFF;">Tips</h2>
        <ul style="color: #FFFFFF;">
            <li>Larger PDFs may take longer to process</li>
            <li>Keep highlights clean and distinct for better extraction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if page == "Upload PDF":
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        try:
            # Extract highlights
            highlights = extract_highlights(pdf_path)
            
            # Display highlights
            if highlights:
                st.subheader(f"Found {len(highlights)} highlights:")
                
                # Save button without columns
                if st.button("💾 Save Highlights", type="primary", key="save_highlights_btn"):
                    doc_id = save_highlights(uploaded_file.name, highlights)
                    st.success(f"Successfully saved {len(highlights)} highlights!")
                    # Reset topic model to include new document
                    st.session_state.topics = None
                
                # Display highlights
                for i, highlight in enumerate(highlights, 1):
                    with st.expander(f"Highlight {i} (Page {highlight['page']})"):
                        st.write(highlight['text'])
            else:
                st.warning("No highlights found in the PDF. Make sure your PDF contains highlighted text.")
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(pdf_path)
    else:
        st.info("Please upload a PDF file to extract highlights.")

else:  # View Saved Highlights page
    st.subheader("📚 Saved Highlights")
    view_saved_highlights() 