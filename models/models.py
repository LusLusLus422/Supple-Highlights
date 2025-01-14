"""Database models for the document management system.

This module defines the SQLAlchemy ORM models used for storing documents,
highlights, and watched directories. It provides the core data structure
for managing PDF documents and their associated highlights.

Classes:
    Document: Represents a PDF document in the system
    Highlight: Represents a highlighted section within a document
    WatchedDirectory: Represents a directory being monitored for new documents
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .base import Base
from datetime import datetime

class Document(Base):
    """Represents a PDF document in the system.
    
    This model stores metadata about PDF documents including their location
    on disk and creation/modification timestamps.
    
    Attributes:
        id (int): Primary key identifier
        filename (str): Name of the PDF file
        filepath (str): Full path to the PDF file on disk
        title (str): Document title
        created_at (datetime): Timestamp when record was created
        updated_at (datetime): Timestamp when record was last updated
        highlights (List[Highlight]): Associated highlight objects
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String, unique=True)
    title = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    highlights = relationship("Highlight", back_populates="document", cascade="all, delete-orphan")

class Highlight(Base):
    """Represents a highlighted section within a PDF document.
    
    Stores the location and content of highlights made by users in PDF documents.
    
    Attributes:
        id (int): Primary key identifier
        document_id (int): Foreign key to parent document
        text (str): The highlighted text content
        page_number (int): Page number where highlight appears
        rect_x0 (float): Left X coordinate of highlight rectangle
        rect_y0 (float): Bottom Y coordinate of highlight rectangle
        rect_x1 (float): Right X coordinate of highlight rectangle
        rect_y1 (float): Top Y coordinate of highlight rectangle
        created_at (datetime): Timestamp when record was created
        updated_at (datetime): Timestamp when record was last updated
        document (Document): Reference to parent document object
    """
    __tablename__ = "highlights"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(String)
    page_number = Column(Integer)
    rect_x0 = Column(Float)
    rect_y0 = Column(Float)
    rect_x1 = Column(Float)
    rect_y1 = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    document = relationship("Document", back_populates="highlights")

class WatchedDirectory(Base):
    """Represents a directory being monitored for new documents.
    
    Tracks directories that should be scanned periodically for new PDF documents
    to import into the system.
    
    Attributes:
        id (int): Primary key identifier
        path (str): Full filesystem path to watched directory
        last_scan (datetime): Timestamp of last directory scan
        created_at (datetime): Timestamp when record was created
    """
    __tablename__ = 'watched_directories'
    
    id = Column(Integer, primary_key=True)
    path = Column(String(500), nullable=False, unique=True)
    last_scan = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<WatchedDirectory(path='{self.path}')>"