import streamlit as st
import plotly.express as px
from pyvis.network import Network
import networkx as nx
from wordcloud import WordCloud
import tempfile
import os
from typing import List
import pandas as pd

from pdf_processor import PDFProcessor
from nlp_pipeline import NLPPipeline
from models.base import Session
from models.models import Document, Highlight, Topic

def create_knowledge_graph(highlights: List[Highlight]) -> str:
    """Create and save an interactive knowledge graph."""
    G = nx.Graph()
    
    # Add nodes for documents and topics
    for highlight in highlights:
        if highlight.document and highlight.topic:
            G.add_node(f"doc_{highlight.document.id}", 
                      label=highlight.document.title, 
                      group=1)
            G.add_node(f"topic_{highlight.topic.id}", 
                      label=highlight.topic.name, 
                      group=2)
            G.add_edge(f"doc_{highlight.document.id}", 
                      f"topic_{highlight.topic.id}")
    
    # Convert to PyVis network
    net = Network(notebook=True, width="100%", height="600px")
    net.from_nx(G)
    
    # Save to HTML file
    html_path = "static/knowledge_graph.html"
    net.save_graph(html_path)
    return html_path

def create_word_cloud(topics: List[Topic]) -> WordCloud:
    """Create word cloud from topics."""
    text = " ".join([topic.description for topic in topics])
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white').generate(text)
    return wordcloud

def main():
    st.title("Supple Highlights Dashboard")
    
    # Initialize session
    session = Session()
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process PDF
        with st.spinner("Processing PDF..."):
            processor = PDFProcessor()
            doc = processor.process_pdf(tmp_path)
            processor.close()
        
        # Run topic modeling
        with st.spinner("Analyzing topics..."):
            nlp = NLPPipeline()
            nlp.create_topic_model()
            nlp.close()
        
        os.unlink(tmp_path)
        st.success("PDF processed successfully!")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Topics", "Highlights"])
    
    with tab1:
        st.header("Knowledge Graph")
        highlights = session.query(Highlight).all()
        if highlights:
            graph_path = create_knowledge_graph(highlights)
            with open(graph_path, 'r') as f:
                html = f.read()
            st.components.v1.html(html, height=600)
    
    with tab2:
        st.header("Topic Explorer")
        topics = session.query(Topic).all()
        if topics:
            wordcloud = create_word_cloud(topics)
            st.image(wordcloud.to_array())
            
            # Topic details
            for topic in topics:
                with st.expander(f"Topic: {topic.name}"):
                    st.write(topic.description)
                    highlights = session.query(Highlight).filter_by(topic_id=topic.id).all()
                    for highlight in highlights:
                        st.text(f"From: {highlight.document.title}")
                        st.write(highlight.text)
                        st.divider()
    
    with tab3:
        st.header("All Highlights")
        documents = session.query(Document).all()
        if documents:
            selected_doc = st.selectbox(
                "Select Document",
                options=documents,
                format_func=lambda x: x.title
            )
            
            highlights = session.query(Highlight).filter_by(document_id=selected_doc.id).all()
            for highlight in highlights:
                with st.expander(f"Page {highlight.page_number}"):
                    st.write(highlight.text)
                    if highlight.topic:
                        st.caption(f"Topic: {highlight.topic.name}")
    
    session.close()

if __name__ == "__main__":
    main() 