import fitz  # PyMuPDF
from typing import List, Dict, Any
import os
from models.models import Document, Highlight
from models.base import Session

class PDFProcessor:
    def __init__(self):
        self.session = Session()
    
    def process_pdf(self, filepath: str) -> Document:
        """Process a PDF file and extract highlights."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        # Create document record
        filename = os.path.basename(filepath)
        doc_record = Document(
            title=filename,
            filepath=filepath
        )
        self.session.add(doc_record)
        
        # Open PDF and extract highlights
        pdf_doc = fitz.open(filepath)
        highlights = []
        
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            annots = page.annots()
            
            if annots:
                for annot in annots:
                    if annot.type[0] == 8:  # Highlight annotation
                        highlight_text = self._extract_highlight_text(page, annot)
                        if highlight_text:
                            highlight = Highlight(
                                document=doc_record,
                                text=highlight_text,
                                page_number=page_num + 1
                            )
                            highlights.append(highlight)
        
        self.session.add_all(highlights)
        self.session.commit()
        return doc_record
    
    def _extract_highlight_text(self, page: fitz.Page, annot: fitz.Annot) -> str:
        """Extract text from a highlight annotation."""
        points = annot.vertices
        if not points:
            return ""
        
        # Get highlighted text
        quad_count = int(len(points) / 4)
        sentences = []
        for i in range(quad_count):
            # Convert quad points to rectangle
            r = fitz.Quad(points[i * 4 : i * 4 + 4]).rect
            words = page.get_text("text", clip=r).strip()
            sentences.append(words)
        
        return " ".join(sentences)
    
    def close(self):
        """Close the database session."""
        self.session.close() 