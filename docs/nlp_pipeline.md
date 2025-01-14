# Enhanced NLP Pipeline Documentation

The enhanced NLP pipeline provides advanced text analysis capabilities including language detection, translation, topic modeling, and semantic search. This document explains how to use these features effectively.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the LLaMA model:
```bash
python scripts/setup_llama.py
```

3. Initialize the database:
```bash
alembic upgrade head
```

## Core Features

### 1. Language Detection and Translation

The pipeline automatically detects the language of highlights and translates non-English text to English for processing while preserving the original text.

```python
from app.nlp_pipeline import NLPPipeline
from models.models import Highlight

# Initialize pipeline
pipeline = NLPPipeline()

# Process a highlight
highlight = Highlight(text="Bonjour le monde!", page_number=1)
pipeline.process_highlight(highlight)

print(f"Original Language: {highlight.original_language}")  # 'fr'
print(f"Translation: {highlight.translated_text}")  # 'Hello world!'
```

### 2. Topic Modeling

The pipeline uses BERTopic for advanced topic modeling, with LLaMA providing natural language topic names.

```python
# Process multiple highlights
highlights = [
    Highlight(text="Machine learning models need data.", page_number=1),
    Highlight(text="Neural networks learn patterns.", page_number=1),
    Highlight(text="GPUs accelerate AI training.", page_number=1)
]

for h in highlights:
    pipeline.process_highlight(h)

# Update topics
pipeline.update_topics(min_highlights=3)

# Access topics
for highlight in highlights:
    if highlight.topic:
        print(f"Text: {highlight.text}")
        print(f"Topic: {highlight.topic.name}")
        print(f"Confidence: {highlight.topic_confidence}")
        print(f"Keywords: {highlight.topic.keywords}")
```

### 3. Semantic Search

The pipeline enables semantic search across highlights using BERT embeddings.

```python
# Search for semantically similar highlights
results = pipeline.semantic_search(
    query="artificial intelligence and deep learning",
    limit=5,
    min_similarity=0.5
)

for highlight, similarity in results:
    print(f"Text: {highlight.text}")
    print(f"Similarity: {similarity:.2f}")
```

## Best Practices

1. **Language Handling**:
   - The pipeline automatically handles non-English text
   - Original text is preserved while translations are used for processing
   - No special handling needed for mixed-language documents

2. **Topic Modeling**:
   - Process at least 5 highlights before running topic modeling
   - Topics are hierarchical and can have parent-child relationships
   - Topic confidence scores help identify strong topic assignments

3. **Semantic Search**:
   - Queries can be in natural language
   - Adjust `min_similarity` based on your needs (0.5 is a good starting point)
   - Results are sorted by relevance

## Performance Considerations

1. **Memory Usage**:
   - LLaMA model requires ~4GB RAM
   - Embeddings are stored in binary format to save space
   - Topic modeling is memory-intensive with large numbers of highlights

2. **Processing Speed**:
   - Language detection and translation are relatively fast
   - Topic modeling becomes slower with more highlights
   - Semantic search scales well due to pre-computed embeddings

3. **Optimization Tips**:
   - Process highlights in batches
   - Update topics periodically rather than after each new highlight
   - Consider cleaning up old embeddings for deleted highlights

## Error Handling

The pipeline includes robust error handling for common scenarios:

```python
try:
    pipeline.process_highlight(highlight)
except Exception as e:
    print(f"Error processing highlight: {e}")

try:
    pipeline.update_topics()
except Exception as e:
    print(f"Error updating topics: {e}")
```

## Integration with Streamlit UI

The pipeline integrates seamlessly with the Streamlit interface:

```python
import streamlit as st
from app.nlp_pipeline import NLPPipeline

pipeline = NLPPipeline()

# Display topics
topics = session.query(Topic).all()
for topic in topics:
    st.write(f"Topic: {topic.name}")
    st.write(f"Keywords: {', '.join(topic.keywords.keys())}")
    
# Search interface
query = st.text_input("Search highlights:")
if query:
    results = pipeline.semantic_search(query)
    for highlight, score in results:
        st.write(f"Match ({score:.2f}): {highlight.text}")
``` 