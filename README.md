# Supple Highlights

An intelligent PDF highlight management system that helps you organize, analyze, and visualize text highlights from your documents using advanced NLP techniques.

## Key Features

- **Smart Highlight Extraction**
  - Automatically extract highlighted text from PDF documents using PyMuPDF
  - Preserve context, page numbers, and spatial information
  - Support for multiple highlight colors and annotation types

- **Advanced Text Analysis**
  - Topic modeling using BERTopic with UMAP and HDBSCAN
  - Multi-language support with automatic translation
  - Semantic search using sentence transformers
  - Optional LLaMA integration for enhanced topic naming

### Text Modeling Process

The text analysis pipeline employs a sophisticated multi-stage approach:

1. **Text Preprocessing**
   - Unicode normalization and special character removal
   - Language detection using `langdetect`
   - Automatic translation to English for non-English text using Google Translate
   - Text cleaning and standardization

2. **Semantic Embedding**
   - Utilizes BAAI/bge-small-en-v1.5 transformer model
   - Generates high-dimensional vector representations
   - Enables semantic similarity comparisons
   - Powers the semantic search functionality

3. **Topic Modeling with BERTopic**
   - Dimensionality reduction using UMAP
   - Density-based clustering with HDBSCAN
   - Topic extraction from highlight clusters
   - Keyword extraction and scoring
   - Dynamic topic number determination

4. **Topic Naming**
   - Initial keyword extraction from topic clusters
   - LLaMA-powered natural language topic name generation
   - Fallback to keyword-based naming when needed
   - Ensures clear, concise, and meaningful topic labels

The pipeline processes highlights incrementally and updates topics dynamically as new content is added, maintaining a coherent and evolving topic structure across your highlights.

- **Interactive Visualizations**
  - Dynamic knowledge graphs showing document relationships
  - Interactive topic visualization with word clouds
  - Highlight distribution analysis
  - Custom visualization themes

- **Efficient Data Management**
  - SQLite database with SQLAlchemy ORM
  - Automatic directory monitoring for PDF changes
  - Full-text search capabilities
  - Document and highlight metadata tracking

## System Requirements

- Python 3.11 or higher
- 4GB RAM minimum (8GB recommended)
- Operating Systems:
  - macOS 10.15+
  - Ubuntu 20.04+ or other modern Linux
  - Windows 10/11

## System Architecture

### Component Overview
1. **Frontend Layer (Streamlit)**
   - Handles user interface and interactions
   - Renders visualizations and data displays

2. **Processing Layer**
   - PDF Processor (PyMuPDF) for highlight extraction
   - NLP Pipeline (BERTopic, NLTK) for text analysis
   - Directory Scanner for file monitoring
   - Optional LLaMA integration for advanced features

3. **Data Layer**
   - SQLite Database with SQLAlchemy ORM
   - Alembic for database migrations
   - Efficient file system operations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd supple_highlights
```

2. Create and activate a virtual environment:
```bash
# macOS/Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r dev-requirements.txt
```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet'])"
```

Note: The SQLite database will be automatically initialized when you first run the application.

## Quick Start

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Set up a watched directory for your PDF files

4. Upload or place PDF documents with highlights in the watched directory

5. Explore your highlights through:
   - Topic visualization
   - Knowledge graphs
   - Search functionality
   - Custom filters

## Project Structure

```
supple_highlights/
├── app/                    # Application modules
│   ├── __init__.py
│   ├── directory_scanner.py # File system monitoring
│   ├── nlp_pipeline.py     # Text analysis and topic modeling
│   ├── pdf_processor.py    # PDF processing and extraction
│   └── models/            # Database models
├── models/                # ML models directory
│   └── llama/            # Optional LLaMA model files
├── migrations/           # Alembic database migrations
├── tests/               # Test suite
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── app.py              # Main application entry point
├── requirements.txt    # Core dependencies
└── dev-requirements.txt # Development dependencies
```

## Development

### Setting Up Development Environment

1. Install all dependencies including development tools:
```bash
pip install -r requirements.txt -r dev-requirements.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Code Quality

- Run tests:
```bash
pytest tests/
```

- Format code:
```bash
black .
isort .
```

- Check code quality:
```bash
flake8
```

## Performance Considerations

### Memory Management
- Chunk-based processing for large PDF files
- Efficient model caching for NLP operations
- Lazy loading of visualizations

### Database Optimization
- Indexed fields for fast queries
- Proper session handling and connection pooling
- Atomic operations with error recovery

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyMuPDF for PDF processing
- BERTopic for advanced topic modeling
- Streamlit for the web interface
- The open-source community for various dependencies 