# Supple Highlights

An intelligent PDF highlight management system that helps you organize, analyze, and visualize text highlights from your documents using advanced NLP techniques.

## Key Features

- **Smart Highlight Extraction**
  - Automatically extract highlighted text from PDF documents
  - Preserve context, page numbers, and spatial information
  - Support for multiple highlight colors and annotation types

- **Advanced Text Analysis**
  - Topic modeling using LDA (Latent Dirichlet Allocation)
  - Semantic similarity analysis
  - Automatic keyword extraction and categorization
  - Natural language processing with NLTK and transformers

- **Interactive Visualizations**
  - Dynamic knowledge graphs showing document relationships
  - Interactive topic visualization with word clouds
  - Highlight distribution analysis
  - Custom visualization themes

- **Efficient Data Management**
  - SQLite database for reliable data storage
  - Full-text search capabilities
  - Document and highlight metadata tracking
  - Export functionality for backup and sharing

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
   - NLP Pipeline (NLTK, Gensim) for text analysis
   - Visualization Engine (PyVis, WordCloud)

3. **Data Layer**
   - SQLite Database with SQLAlchemy ORM
   - Efficient file system operations
   - Memory-optimized data processing

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

## Quick Start

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload a PDF document with highlights

4. Explore your highlights through:
   - Topic visualization
   - Knowledge graphs
   - Search functionality
   - Custom filters

## Project Structure

```
supple_highlights/
├── app/                    # Application modules
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF processing and extraction
│   ├── nlp_pipeline.py    # NLP and topic modeling
│   └── directory_scanner.py # File system operations
├── models/                 # Database components
│   ├── base.py            # SQLAlchemy configuration
│   └── models.py          # Data models and schemas
├── tests/                 # Test suite
├── app.py                 # Main application entry point
├── requirements.txt       # Core dependencies
└── dev-requirements.txt   # Development dependencies
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
- Streamlit for the web interface
- NLTK and scikit-learn for NLP capabilities
- The open-source community for various dependencies 