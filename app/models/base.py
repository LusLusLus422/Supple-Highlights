"""SQLAlchemy base configuration.

This module provides the base SQLAlchemy setup including:
- Base class for declarative models
- Database engine configuration
- Session factory
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session as SQLAlchemySession
from contextlib import contextmanager

# Create database engine with better concurrency handling
engine = create_engine(
    'sqlite:///app.db',
    connect_args={
        'timeout': 30,  # Increase SQLite timeout
        'check_same_thread': False  # Allow multi-threading
    },
    pool_size=20,  # Connection pool size
    max_overflow=0  # Max number of connections to overflow
)

# Create declarative base
Base = declarative_base()

# Create session factory
Session = sessionmaker(bind=engine)
SessionLocal = Session  # Add alias for compatibility

@contextmanager
def get_db_session() -> SQLAlchemySession:
    """Get a database session using a context manager.
    
    This ensures the session is properly closed after use.
    
    Yields:
        SQLAlchemy Session object
    """
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_session() -> SQLAlchemySession:
    """Get a new database session.
    
    Returns:
        SQLAlchemy Session object
    """
    return Session() 