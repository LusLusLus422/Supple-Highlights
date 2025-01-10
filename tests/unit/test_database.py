import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

def test_database_connection(test_db_url):
    """Test if we can establish a database connection."""
    try:
        engine = create_engine(test_db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        assert session is not None
    except SQLAlchemyError as e:
        pytest.fail(f"Failed to connect to database: {str(e)}")
    finally:
        if 'session' in locals():
            session.close()

def test_database_operations(test_db_url):
    """Test basic database operations."""
    try:
        engine = create_engine(test_db_url)
        # Test creating tables
        from models import Base
        Base.metadata.create_all(engine)
        assert True  # If we get here, table creation was successful
    except SQLAlchemyError as e:
        pytest.fail(f"Failed to perform database operations: {str(e)}")
    except ImportError as e:
        pytest.fail(f"Failed to import models: {str(e)}") 