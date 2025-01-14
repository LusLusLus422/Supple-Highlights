"""Directory scanning module for PDF highlight management.

This module provides functionality to scan and monitor directories for PDF files,
extract highlights from them, and manage the persistence of these highlights in
the database. It handles both initial scanning and subsequent updates to track
changes in PDF highlights over time.

Typical usage example:
    scanner = DirectoryScanner()
    scanner.set_watched_directory("/path/to/pdfs")
    documents = scanner.scan_directory()
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from models.models import Document, Highlight, WatchedDirectory
from models.base import SessionLocal
from .pdf_processor import PDFProcessor

class DirectoryScanner:
    """Manages directory scanning and highlight extraction from PDF files.
    
    This class handles the monitoring of directories containing PDF files,
    extracting highlights from these files, and managing the persistence
    of these highlights in the database.
    """

    def __init__(self):
        """Initialize DirectoryScanner with database session and PDF processor."""
        self.session = SessionLocal()
        self.pdf_processor = PDFProcessor()

    def get_watched_directory(self) -> Optional[WatchedDirectory]:
        """Retrieve the currently configured watched directory.

        Returns:
            Optional[WatchedDirectory]: The currently watched directory configuration
                or None if no directory is configured.
        """
        return self.session.query(WatchedDirectory).first()

    def set_watched_directory(self, path: str) -> WatchedDirectory:
        """Set or update the directory to be watched for PDF files.

        Args:
            path (str): The filesystem path to watch for PDF files.

        Returns:
            WatchedDirectory: The created or updated watched directory configuration.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not os.path.isdir(path):
            raise ValueError(f"Invalid directory path: {path}")

        watched_dir = self.get_watched_directory()
        if watched_dir:
            watched_dir.path = path
        else:
            watched_dir = WatchedDirectory(path=path)
            self.session.add(watched_dir)
        
        self.session.commit()
        return watched_dir

    def update_document_highlights(self, doc: Document, filepath: str) -> bool:
        """Update highlights for a document, only processing changes.

        Extracts highlights from the PDF and compares them with existing ones,
        adding new highlights and removing deleted ones.

        Args:
            doc (Document): The document object to update highlights for.
            filepath (str): Path to the PDF file to process.

        Returns:
            bool: True if any changes were made, False otherwise.
        """
        # Extract current highlights from PDF
        new_highlights_data = self.pdf_processor.extract_highlights_from_pdf(filepath)
        
        # Create sets for comparison using immutable tuples
        existing_highlights = {
            (h.text, h.page_number, h.rect_x0, h.rect_y0, h.rect_x1, h.rect_y1)
            for h in doc.highlights
        }
        
        new_highlights = {
            (h['text'], h['page_number'], h['rect_x0'], h['rect_y0'], h['rect_x1'], h['rect_y1'])
            for h in new_highlights_data
        }
        
        # Find highlights to add and remove
        highlights_to_remove = existing_highlights - new_highlights
        highlights_to_add = new_highlights - existing_highlights
        
        if not highlights_to_add and not highlights_to_remove:
            return False  # No changes needed
        
        # Remove deleted highlights
        for highlight_tuple in highlights_to_remove:
            text, page_number, x0, y0, x1, y1 = highlight_tuple
            highlight = next(
                h for h in doc.highlights
                if (h.text == text and h.page_number == page_number and
                    h.rect_x0 == x0 and h.rect_y0 == y0 and
                    h.rect_x1 == x1 and h.rect_y1 == y1)
            )
            self.session.delete(highlight)
        
        # Add new highlights
        current_time = datetime.utcnow()
        for highlight_tuple in highlights_to_add:
            text, page_number, x0, y0, x1, y1 = highlight_tuple
            new_highlight = Highlight(
                document=doc,
                text=text,
                page_number=page_number,
                rect_x0=x0,
                rect_y0=y0,
                rect_x1=x1,
                rect_y1=y1,
                created_at=current_time,
                updated_at=current_time
            )
            self.session.add(new_highlight)
        
        # Update document timestamp
        doc.updated_at = current_time
        
        self.session.commit()
        return True

    def scan_directory(self, callback=None) -> List[Document]:
        """Scan the watched directory and process PDF highlights.

        Scans the configured directory for PDF files, processes their highlights,
        and updates the database accordingly.

        Args:
            callback (Optional[Callable[[str, int], None]]): Optional progress callback
                function that receives filename and count parameters.

        Returns:
            List[Document]: List of processed documents.

        Raises:
            ValueError: If no watched directory is configured.
        """
        watched_dir = self.get_watched_directory()
        if not watched_dir:
            raise ValueError("No watched directory configured")

        processed_docs = []
        total_pdfs = 0
        
        # First, count total PDFs for progress calculation
        for root, _, files in os.walk(watched_dir.path):
            total_pdfs += len([f for f in files if f.lower().endswith('.pdf')])
        
        processed_count = 0
        for root, _, files in os.walk(watched_dir.path):
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            
            for pdf_file in pdf_files:
                filepath = os.path.join(root, pdf_file)
                file_mtime = os.path.getmtime(filepath)
                processed_count += 1
                
                try:
                    # Check if file exists in database
                    existing_doc = self.session.query(Document).filter_by(filepath=filepath).first()
                    
                    if not existing_doc:
                        # New file - process normally
                        doc = self.pdf_processor.process_pdf(filepath)
                        processed_docs.append(doc)
                        if callback:
                            callback(pdf_file, processed_count)
                    elif file_mtime > watched_dir.last_scan.timestamp():
                        # Modified file - update only changed highlights
                        if self.update_document_highlights(existing_doc, filepath):
                            processed_docs.append(existing_doc)
                            if callback:
                                callback(f"Updated: {pdf_file}", processed_count)
                
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")

        # Update last scan timestamp
        watched_dir.last_scan = datetime.utcnow()
        self.session.commit()

        return processed_docs 