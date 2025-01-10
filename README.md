# Supple Highlights

A tool for extracting and visualizing text highlights from books and articles.

## Features

- Extract highlights from PDF documents
- Process and analyze text using NLP techniques
- Visualize highlights and their relationships
- Store and manage highlights in a database

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

4. Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
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

## Testing

Run tests with pytest:
```bash
pytest tests/
```

For coverage report:
```bash
pytest --cov=supple_highlights tests/
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Upload a PDF document
3. View and manage highlights
4. Explore visualizations

## Project Structure

```
supple_highlights/
├── app.py                 # Main Streamlit application
├── models/               # Database models
├── processors/          # Text processing modules
├── visualizers/         # Visualization modules
├── tests/              # Test suite
│   ├── unit/
│   └── integration/
└── requirements.txt    # Project dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 