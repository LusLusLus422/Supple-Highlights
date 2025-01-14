"""PDF Processing Module.

This module handles the extraction and processing of PDF documents and their highlights.
It provides functionality to:
- Extract highlighted text from PDF files
- Process PDFs and store their information in the database
- Manage highlight annotations and their associated metadata

The module uses PyMuPDF (fitz) for PDF processing and integrates with the application's
database models for data persistence.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Any
import os
from app.models.models import Document, Highlight
from app.models.base import SessionLocal
from datetime import datetime
import re

class PDFProcessor:
    """Handles PDF document processing and highlight extraction.
    
    This class provides methods to process PDF files, extract highlights,
    and manage the associated database records.
    """

    def __init__(self):
        """Initialize PDFProcessor with a database session."""
        self.session = SessionLocal()
    
    def extract_highlights_from_pdf(self, filepath: str) -> List[Dict]:
        """Extract all highlights from a PDF file without saving to database.
        
        Args:
            filepath (str): Path to the PDF file to process
            
        Returns:
            List[Dict]: List of dictionaries containing highlight data with keys:
                       'text', 'page_number', 'rect_x0', 'rect_y0', 'rect_x1', 'rect_y1'
                       
        Raises:
            FileNotFoundError: If the specified PDF file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        highlights_data = []
        pdf_doc = fitz.open(filepath)
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            annots = page.annots()
            
            if annots:
                for annot in annots:
                    if annot.type[0] == 8:  # Highlight annotation
                        text = self._extract_highlight_text(page, annot)
                        if text:
                            highlights_data.append({
                                'text': text,
                                'page_number': page_num + 1,
                                'rect_x0': annot.rect.x0,
                                'rect_y0': annot.rect.y0,
                                'rect_x1': annot.rect.x1,
                                'rect_y1': annot.rect.y1
                            })
        
        pdf_doc.close()
        return highlights_data
    
    def process_pdf(self, filepath: str) -> Document:
        """Process a PDF file and create database records for document and highlights.
        
        Args:
            filepath (str): Path to the PDF file to process
            
        Returns:
            Document: Created document database record with associated highlights
            
        Raises:
            FileNotFoundError: If the specified PDF file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        # Create document record
        current_time = datetime.utcnow()
        filename = os.path.basename(filepath)
        doc_record = Document(
            title=filename,
            filename=filename,
            filepath=filepath,
            created_at=current_time,
            updated_at=current_time
        )
        self.session.add(doc_record)
        
        # Extract and create highlights
        highlights_data = self.extract_highlights_from_pdf(filepath)
        highlights = [
            Highlight(
                document=doc_record,
                created_at=current_time,
                updated_at=current_time,
                **highlight_data
            )
            for highlight_data in highlights_data
        ]
        
        self.session.add_all(highlights)
        self.session.commit()
        return doc_record
    
    def _extract_highlight_text(self, page: fitz.Page, annot: fitz.Annot) -> str:
        """Extract text content from a highlight annotation on a PDF page.
        
        Args:
            page (fitz.Page): PDF page containing the highlight
            annot (fitz.Annot): Highlight annotation object
            
        Returns:
            str: Extracted and cleaned text from the highlighted area
                 Returns empty string if extraction fails
        """
        points = annot.vertices
        if points:
            try:
                # Extract text from the highlighted rectangle area
                rect = annot.rect
                text = page.get_text("text", clip=rect)
                
                # Clean up the extracted text
                text = text.strip()
                
                # Remove HTML-like content and entities
                text = re.sub(r'<[^>]*>|</[^>]*>', '', text)  # Remove HTML tags
                text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)   # Remove HTML entities
                
                # Normalize whitespace and return
                return ' '.join(text.split())
            except Exception as e:
                print(f"Error extracting highlight text: {str(e)}")
                return ""
        return ""
    
    def close(self):
        """Close the database session and free resources."""
        self.session.close()