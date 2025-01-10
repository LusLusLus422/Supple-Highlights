from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    title = Column(String)
    created_at = Column(DateTime)
    highlights = relationship("Highlight", back_populates="document", cascade="all, delete-orphan")

class Highlight(Base):
    __tablename__ = "highlights"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(String)
    page_number = Column(Integer)
    rect_x0 = Column(Float)
    rect_y0 = Column(Float)
    rect_x1 = Column(Float)
    rect_y1 = Column(Float)
    document = relationship("Document", back_populates="highlights") 