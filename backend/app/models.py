# from sqlalchemy import Column, Integer, String, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from datetime import datetime

# Base = declarative_base()

# class Document(Base):
#     __tablename__ = "Document"

#     id = Column(Integer, primary_key=True, index=True)
#     filename = Column(String, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     content = Column(String)  # Store extracted text content here

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True, nullable=False)
    file_path = Column(String(512), nullable=False)
    content = Column(Text)  # Using Text for large content
    upload_date = Column(DateTime, default=datetime.utcnow)