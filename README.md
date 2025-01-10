# Supple Highlights

A tool for extracting and visualizing text highlights from books and articles.

## Features

- Extract highlights from PDF documents with PyMuPDF
- Process and analyze text using NLP techniques (NLTK, scikit-learn)
- Generate interactive knowledge graphs and topic visualizations
- Store and manage highlights in SQLite database
- Modern Streamlit interface with search and filtering capabilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd supple_highlights
```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Unix/macOS
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install required NLTK data:
```bash
python -m nltk.downloader punkt stopwords wordnet
```

## Development Setup

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Format code:
```bash
black .
isort .
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Upload a PDF document with highlighted text
3. View and manage extracted highlights
4. Explore topic modeling and knowledge graph visualizations
5. Search and filter across all your saved highlights

## Project Structure

```
supple_highlights/
├── app/                  # Application modules
│   ├── main.py          # Core application logic
│   └── nlp_pipeline.py  # NLP processing functionality
├── app.py               # Main Streamlit application entry point
├── models/              # Database models and configuration
│   ├── base.py         # SQLAlchemy setup
│   └── models.py       # Data models
├── static/             # Static assets
├── templates/          # HTML templates
├── DETAILED_GUIDE.md   # Detailed usage instructions
├── Project Outline.md  # Project planning and architecture
├── requirements.txt    # Project dependencies
└── LICENSE            # MIT License
```

## Features in Detail

### PDF Processing
- Extracts highlighted text from PDF documents
- Preserves page numbers and highlight positions
- Supports multiple highlights per page

### Topic Modeling
- Automatic topic extraction from highlights
- Interactive topic visualization with word clouds
- Topic strength analysis

### Knowledge Graph
- Interactive network visualization
- Document-topic relationships
- Customizable graph layout and styling

### Search and Organization
- Full-text search across all highlights
- Filter by document or topic
- Chronological organization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 