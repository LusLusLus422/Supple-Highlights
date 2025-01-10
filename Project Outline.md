**Supple Highlights: Explorative Personalized Knowledge Dashboard**

Transform the way you read, analyze, and retain information. **Supple Highlights** extracts, categorizes, and recontextualizes text highlights from books and articles, empowering you to:

•	**Spot Patterns**: Dive deep into automated clustering and topic modeling to uncover connections across your readings.

•	**Organize Seamlessly**: Tag, group, and classify highlights with a user-driven dashboard that adapts to your knowledge needs.

•	**Recontextualize Insights**: Explore new perspectives through dynamic topic suggestions and smart summaries.

Powered by Python and SQLite, with cutting-edge unsupervised learning models, **Supple Highlights** isn’t just a tool—it’s your partner in personal knowledge management.

## Business Problem

- Automated categorisation of texts Aggregated by user
- Explorative Tool for Recontextualising and Aggregating information
- Personal Knowledge Management

## DS & Model

    
    Here’s a **super simple tech stack** for the backend **after ML integration** and with a **basic dashboard implementation**:
    
    **Tech Stack**
    
    **Backend:**
    
    1.	**Language**: Python
    
    2.	**Framework**: Flask or FastAPI (for REST API and dashboard backend)
    
    3.	**PDF/Text Extraction**: PyMuPDF (fitz) for extracting highlights.
    
    4.	**Database**: SQLite to store and retrieve data.
    
    5.	**ML Library**: Scikit-learn for clustering and topic modeling.
    
    **Frontend (Dashboard):**
    
    1.	**Framework**: Streamlit or Dash for a lightweight, interactive dashboard.
    
    2.	**Charting Library**: Plotly for visualizations like topic distributions or clustering results.
    
    **Architecture**
    
    **1. Data Flow**
    
    1.	**Data Extraction**:
    
    •	Extract highlights and metadata from PDFs using PyMuPDF.
    
    •	Store in SQLite with fields like:
    
    •	id, title, author, highlight, page, link, topic_id.
    
    2.	**ML Pipeline**:
    
    •	Use Scikit-learn for clustering or topic modeling:
    
    •	Preprocess text (cleaning, tokenizing).
    
    •	Apply TF-IDF vectorization for feature extraction.
    
    •	Use K-Means or LDA to group highlights.
    
    •	Save the clustering/topic results (topic_id or cluster_id) in SQLite.
    
    3.	**API**:
    
    •	Create Flask/FastAPI endpoints:
    
    •	GET /highlights: Fetch all highlights.
    
    •	GET /topics: Fetch grouped highlights.
    
    •	POST /upload: Upload new documents for processing.
    
    •	POST /recluster: Trigger re-clustering or topic modeling.
    
    **2. Simple Dashboard**
    
    •	Use **Streamlit** or **Dash** for a no-frills web app interface:
    
    •	Fetch data from the Flask/FastAPI backend.
    
    •	Display it interactively with minimal setup.
    
    **Dashboard Features**
    
    1.	**Highlight Overview**:
    
    •	A table displaying highlights with columns for title, author, page, and text.
    
    2.	**Topic/Cluster View**:
    
    •	Grouped lists of highlights by topic/cluster.
    
    •	Option to click on a topic to drill down and see related highlights.
    
    3.	**Visualizations**:
    
    •	Pie chart: Topic distribution.
    
    •	Scatter plot: Highlights clustered in a 2D space using PCA.
    
    4.	**Upload & Re-cluster**:
    
    •	File upload button for adding new documents.
    
    •	A button to trigger re-clustering.
    
    **Why This Setup?**
    
    •	**Simple Yet Functional**: Streamlit/Dash handles dashboards easily with minimal frontend coding.
    
    •	**Modular**: Backend remains separate, so you can upgrade the API or ML model without impacting the UI.
    
    •	**Expandable**: Transition to a more advanced frontend (e.g., React) later without reworking the backend.
    
    This stack ensures your product is usable quickly while staying flexible for future development.
    

### Python Backend

- PymuPDF for extracting Highlights

### SQLite

- Title
- Author
- Highlights
- Page
- Link

### Unsupervised Training

- Unsupervised Text Classification Sklearn
- **Examples:**
    
    **Clustering:** Groups similar texts without predefined labels.
    
    •	Algorithms: K-Means, DBSCAN, Hierarchical Clustering.
    
    **Topic Modeling:** Identifies underlying topics in text.
    
    •	Algorithms: Latent Dirichlet Allocation (LDA), Non-Negative Matrix Factorization (NMF).
    

## Product Solution

- Dashboard App
- Topic Suggestion based on Text Snippets
- User based Decision on permanency of Topics
- Automated Summarisation of Topics/Category



Step-by-Step Methodology
Step 1: Text Preprocessing
	•	Objective: Prepare the raw text for clustering by cleaning and converting it into a numerical format.
	•	Steps:
	1.	Extract Text:
	•	Extract raw text from files (PDFs, DOCX, etc.) using tools like PyPDF2, pdfplumber, or python-docx.
	2.	Clean the Text:
	•	Normalize text:
	•	Convert to lowercase.
	•	Remove special characters, numbers, and unnecessary whitespace.
	•	Remove stopwords (e.g., "the," "and") using libraries like NLTK or spaCy.
	•	Perform stemming or lemmatization (e.g., "running" → "run") for word standardization.
	3.	Optional Text Splitting:
	•	Split the text into smaller sections or paragraphs if the document is large. Use logical boundaries like headings, paragraphs, or bullet points.
	•	Skip this step for smaller documents.
	•	Outcome:
	•	A cleaned and segmented dataset where each section or paragraph is treated as a separate text sample.
Step 2: Text Vectorization
	•	Objective: Convert preprocessed text into numerical vectors suitable for clustering.
	•	Steps:
	1.	Choose a Vectorization Technique:
	•	TF-IDF (Term Frequency-Inverse Document Frequency):
	•	Captures the importance of words relative to the document.
	•	Works well for smaller datasets or when simplicity is preferred.
	•	Use TfidfVectorizer from sklearn.
	•	Word or Sentence Embeddings:
	•	Use pre-trained models like sentence-transformers (e.g., all-MiniLM) to capture semantic meaning.
	•	More effective for capturing contextual relationships in text.
	2.	Generate Vectors:
	•	Apply the chosen method to convert text into a feature matrix.
	•	Example:
	•	TF-IDF: Sparse matrix of term frequencies.
	•	Embeddings: Dense vector representation of sentences.
	•	Outcome:
	1.	A numerical representation of the text suitable for clustering.

Step 3: Clustering
	•	Objective: Group similar text samples into clusters.
	•	Steps:
	1.	Choose a Clustering Algorithm:
	•	K-Means Clustering:
	•	Requires you to specify the number of clusters (k).
	•	Use if you have a rough idea of the number of categories.
	•	Hierarchical Clustering:
	•	Builds a dendrogram to determine natural groupings.
	•	Suitable when the number of clusters is unknown.
	•	DBSCAN (Density-Based Spatial Clustering):
	•	Identifies clusters based on density, useful for noisy data.
	2.	Apply Clustering:
	•	Input the feature matrix from Step 2 into the clustering algorithm.
	•	Assign cluster labels to each text sample.
	•	Outcome:
	1.	Clustered text samples with assigned labels.


Step 4: Summarization (Optional)
	•	Objective: Use an LLM to summarize the content of each cluster into concise insights.
	•	Steps:
	1.	Concatenate Text per Cluster:
	•	Combine all text samples within each cluster into a single input.
	2.	Summarize Using LLM:
	•	Use an LLM (e.g., OpenAI GPT or Hugging Face) to summarize the concatenated text.


Step 5: Visualization
	•	Objective: Present the clustering results and summaries in an accessible format.
	•	Steps:
	1.	Visualize clusters in 2D/3D space using dimensionality reduction techniques:
	•	t-SNE or UMAP: Reduce high-dimensional vectors into 2D for visualization.
	•	Use matplotlib or plotly for  plots.
	2.	Output Results:
	•	Provide the clustered text and summaries in a structured format (e.g., JSON, CSV).
	•	Optional: Create a simple dashboard using Streamlit or Flask.
	•	Outcome:
	1.	Visualized clusters and summaries for better interpretability.

## Implementation Pipeline

### 1. PDF Processing Module
- Use PyMuPDF to extract highlighted text
- Store metadata and highlights in SQLite database
- Handle different highlight colors and annotation types

### 2. NLP Processing Pipeline
1. **Text Preprocessing**
   - NLTK for tokenization and lemmatization
   - Remove stopwords and special characters
   - Create TF-IDF vectors using scikit-learn

2. **Topic Modeling**
   - Use Latent Dirichlet Allocation (LDA) from gensim
   - Generate topic clusters and keywords
   - Store topic assignments in database

3. **Embedding Generation**
   - Use sentence-transformers for semantic text embeddings
   - Generate document and topic embeddings for visualization

### 3. Dashboard Interface
Built with Streamlit, featuring:

1. **Upload Section**
   - PDF upload interface
   - Progress bar for processing
   - Basic document management

2. **Knowledge Graph View**
   - Interactive network graph using Pyvis
   - Nodes: Topics and Documents
   - Edges: Relationships between topics and documents
   - Click interactions to show related highlights

3. **Topic Explorer**
   - Word cloud visualization of topics
   - Click to filter related highlights
   - Topic strength indicators

4. **Search & Filter**
   - Full-text search across highlights
   - Filter by document, topic, or date range
   - Sort and group functionality

## File Structure

## Setup Instructions

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On MacOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize database:
```bash
python -m app.database
```

4. Run application:
```bash
streamlit run app/main.py
```

## Performance Considerations
- Optimized for MacBook Pro M4 Pro
- Utilize multiprocessing for PDF processing
- Batch processing for large documents
- Caching of embeddings and visualization data

## Future Enhancements
1. Export functionality for knowledge graphs
2. Custom topic labeling
3. Document similarity recommendations
4. Integration with note-taking applications
5. Advanced search with semantic similarity