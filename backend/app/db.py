

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base
from sqlalchemy.exc import SQLAlchemyError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    try:
        Base.metadata.drop_all(bind=engine) 
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully.")
    except SQLAlchemyError as e:
        logger.error(f"Error creating tables: {str(e)}", exc_info=True)
        raise Exception("Error initializing the database")

def get_db():
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        db.close()
