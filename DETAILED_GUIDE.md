# Supple Highlights: Detailed User Guide

## Overview

Supple Highlights is an advanced PDF highlight extraction and analysis tool that helps you organize, analyze, and visualize highlighted text from your PDF documents. This guide explains how the app works from start to finish, including the background processes.

## System Architecture

### Component Overview
1. **Frontend Layer (Streamlit)**
   - Handles user interface
   - Manages user interactions
   - Renders visualizations

2. **Processing Layer**
   - PDF Processor (PyMuPDF)
   - NLP Pipeline (NLTK, Gensim)
   - Visualization Engine (PyVis, WordCloud)

3. **Data Layer**
   - SQLite Database
   - SQLAlchemy ORM
   - File System (temporary storage)

## Background Processes

### 1. PDF Upload and Processing

#### User Action
- User uploads PDF file

#### Background Processes
1. **File Handling**
   ```python
   # Creates temporary file
   with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
       tmp_file.write(uploaded_file.getvalue())
   ```

2. **PDF Processing**
   ```python
   # Opens PDF with PyMuPDF
   doc = fitz.open(pdf_path)
   # Iterates through pages
   for page_num in range(len(doc)):
       # Extracts highlights
       annots = page.annots()
   ```

3. **Memory Management**
   - Temporary file creation
   - File cleanup after processing
   - Memory release

### 2. Highlight Extraction

#### User Action
- System automatically processes highlights after upload

#### Background Processes
1. **Annotation Detection**
   ```python
   # Checks for highlight annotations (type 8)
   if annot.type[0] == 8:
       # Gets highlight coordinates
       points = annot.vertices
       # Extracts text within coordinates
       text = page.get_text("text", clip=rect)
   ```

2. **Text Processing**
   - Coordinate calculation
   - Text extraction from regions
   - Character encoding handling

### 3. Database Operations

#### User Action
- User clicks "Save Highlights"

#### Background Processes
1. **Transaction Management**
   ```python
   # Starts database session
   db = SessionLocal()
   try:
       # Creates document record
       doc = Document(filename=filename)
       db.add(doc)
       # Creates highlight records
       for highlight in highlights:
           h = Highlight(document_id=doc.id, text=highlight['text'])
           db.add(h)
       db.commit()
   finally:
       db.close()
   ```

2. **Relationship Management**
   - Foreign key constraints
   - Cascade deletions
   - Index updates

### 4. NLP Pipeline Activation

#### User Action
- User clicks "Generate Topics" or views Topics tab first time

#### Background Processes
1. **Text Preprocessing**
   ```python
   def preprocess_text(text):
       # Lowercase conversion
       text = text.lower()
       # Special character removal
       text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
       # Tokenization
       words = text.split()
       # Stopword removal
       words = [w for w in words if w not in stopwords]
   ```

2. **Topic Modeling**
   ```python
   # Creates document-term matrix
   dictionary = corpora.Dictionary(texts)
   corpus = [dictionary.doc2bow(text) for text in texts]
   
   # Trains LDA model
   lda_model = models.LdaModel(
       corpus,
       num_topics=num_topics,
       id2word=dictionary,
       passes=30
   )
   ```

3. **Model Management**
   - Model caching in session state
   - Memory optimization
   - Result persistence

### 5. Knowledge Graph Generation

#### User Action
- User clicks "Generate Knowledge Graph"

#### Background Processes
1. **Graph Construction**
   ```python
   # Creates network
   G = Network(height="600px", width="100%")
   
   # Adds nodes and edges
   for doc in documents:
       G.add_node(f"doc_{doc.id}")
       for topic_id, weight in doc_topics:
           G.add_edge(f"doc_{doc.id}", f"topic_{topic_id}")
   ```

2. **Layout Calculation**
   - Force-directed layout
   - Node positioning
   - Edge weight calculation

### 6. Search and Filter Operations

#### User Action
- User enters search query

#### Background Processes
1. **Search Processing**
   ```python
   # Filters highlights
   if search_query:
       highlights = [h for h in doc.highlights 
                    if search_query.lower() in h.text.lower()]
   ```

2. **Result Management**
   - Query optimization
   - Result caching
   - Dynamic updates

## Performance Considerations

### Memory Management
1. **PDF Processing**
   - Chunk-based processing for large files
   - Temporary file cleanup
   - Memory release after processing

2. **NLP Operations**
   - Batch processing for large datasets
   - Model caching
   - Resource cleanup

3. **Visualization**
   - Lazy loading of visualizations
   - Component recycling
   - Memory-efficient rendering

### Database Optimization
1. **Query Performance**
   - Indexed fields
   - Relationship optimization
   - Connection pooling

2. **Transaction Management**
   - Atomic operations
   - Proper session handling
   - Error recovery

## Error Handling

### Critical Points
1. **File Operations**
   - File not found
   - Permission issues
   - Corrupt files

2. **NLP Processing**
   - Insufficient data
   - Model failures
   - Resource exhaustion

3. **Database Operations**
   - Connection failures
   - Transaction rollbacks
   - Constraint violations

## Support and Resources

- Check the sidebar for quick tips
- Refer to this guide for detailed information
- Future updates will be documented here 