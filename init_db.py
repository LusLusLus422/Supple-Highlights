from models.base import engine, Base
from models.models import Document, Highlight, WatchedDirectory

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db() 