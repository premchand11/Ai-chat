import os
import logging
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import fitz
from app.models import Document
from app.db import get_db

router = APIRouter()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Upload directory
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory ready at: {UPLOAD_DIR}")

def sanitize_filename(filename: str) -> str:
    # Remove path separators and keep only the basename
    return os.path.basename(filename).replace("..", "")

@router.post("/upload/", response_class=JSONResponse, status_code=200)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF file, extract its text, and store it in the database.
    """
    # Validate file extension (case-insensitive)
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type attempted: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Sanitize and ensure unique filename
    base_filename = sanitize_filename(file.filename)
    name, ext = os.path.splitext(base_filename)
    file_path = os.path.join(UPLOAD_DIR, base_filename)
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(UPLOAD_DIR, f"{name}_{counter}{ext}")
        counter += 1

    # Read file content
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"File saved at: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    # Extract text from PDF
    try:
        content = ""
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                content += page.get_text()
        logger.info(f"Extracted text from PDF: {file_path}")
    except Exception as e:
        logger.error(f"PDF parsing error: {str(e)}\n{traceback.format_exc()}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Could not parse PDF file: {str(e)}")

    # Save to database
    try:
        document = Document(
            filename=os.path.basename(file_path),
            file_path=file_path,
            content=content
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"Document saved to database with ID: {document.id}")
    except Exception as e:
        logger.error(f"Database error: {str(e)}\n{traceback.format_exc()}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    return {
        "document_id": document.id,
        "message": "File uploaded successfully",
        "filename": document.filename
    }





# import os
# import logging
# import traceback
# from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
# from fastapi.responses import JSONResponse
# from sqlalchemy.orm import Session
# import fitz
# from app.models import Document
# from app.db import get_db

# router = APIRouter()

# # Configure detailed logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Create uploads directory in the current working directory
# UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# logger.info(f"Upload directory created at: {UPLOAD_DIR}")

# @router.post("/upload/")
# async def upload_pdf(
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     try:
#         logger.info(f"Starting upload process for file: {file.filename}")
        
#         # Validate file extension
#         if not file.filename.endswith('.pdf'):
#             logger.warning(f"Invalid file type attempted: {file.filename}")
#             return JSONResponse(
#                 status_code=400,
#                 content={"detail": "Only PDF files are allowed"}
#             )

#         # Create a unique filename
#         base_filename = os.path.basename(file.filename)
#         file_path = os.path.join(UPLOAD_DIR, base_filename)
#         counter = 1
#         while os.path.exists(file_path):
#             name, ext = os.path.splitext(base_filename)
#             file_path = os.path.join(UPLOAD_DIR, f"{name}_{counter}{ext}")
#             counter += 1
        
#         logger.debug(f"Generated unique file path: {file_path}")

#         # Read and save file content
#         try:
#             contents = await file.read()
#             logger.debug(f"Successfully read file contents, size: {len(contents)} bytes")
            
#             with open(file_path, "wb") as f:
#                 f.write(contents)
#             logger.info(f"File saved successfully at: {file_path}")
#         except Exception as e:
#             logger.error(f"Error saving file: {str(e)}\n{traceback.format_exc()}")
#             return JSONResponse(
#                 status_code=500,
#                 content={"detail": f"Could not save file: {str(e)}"}
#             )

#         # Extract text content
#         try:
#             content = ""
#             with fitz.open(file_path) as doc:
#                 for page_num, page in enumerate(doc):
#                     content += page.get_text()
#                     logger.debug(f"Extracted text from page {page_num + 1}")
#             logger.info("PDF text extraction completed successfully")
#         except Exception as e:
#             logger.error(f"PDF parsing error: {str(e)}\n{traceback.format_exc()}")
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#                 logger.info(f"Cleaned up file after parsing error: {file_path}")
#             return JSONResponse(
#                 status_code=400,
#                 content={"detail": f"Could not parse PDF file: {str(e)}"}
#             )

#         # Save to database
#         try:
#             document = Document(
#                 filename=os.path.basename(file_path),
#                 file_path=file_path,
#                 content=content
#             )
#             db.add(document)
#             db.commit()
#             db.refresh(document)
#             logger.info(f"Document saved to database with ID: {document.id}")
#         except Exception as e:
#             logger.error(f"Database error: {str(e)}\n{traceback.format_exc()}")
#             if os.path.exists(file_path):
#                 os.remove(file_path)
#                 logger.info(f"Cleaned up file after database error: {file_path}")
#             return JSONResponse(
#                 status_code=500,
#                 content={"detail": f"Database error: {str(e)}"}
#             )

#         return JSONResponse(
#             status_code=200,
#             content={
#                 "document_id": document.id,
#                 "message": "File uploaded successfully",
#                 "filename": document.filename
#             }
#         )

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
#         return JSONResponse(
#             status_code=500,
#             content={"detail": f"Unexpected error: {str(e)}"}
#         )
    



#compatible for other models



# from fastapi import APIRouter, UploadFile, File, HTTPException
# import fitz  # PyMuPDF
# from app.models import Document
# from app.db import SessionLocal

# router = APIRouter()

# @router.post("/upload/")
# async def upload_pdf(file: UploadFile = File(...)):
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="File must be a PDF")
    
#     content = ""
#     with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
#         for page in doc:
#             content += page.get_text()

#     db = SessionLocal()
#     db_document = Document(filename=file.filename, content=content)
#     db.add(db_document)
#     db.commit()
#     db.refresh(db_document)
#     return {"document_id": db_document.id}
# from fastapi import APIRouter, UploadFile, File, HTTPException
# import fitz  # PyMuPDF
# from app.models import Document
# from app.db import SessionLocal
# from sqlalchemy.exc import SQLAlchemyError

# router = APIRouter()

# @router.post("/upload/")
# async def upload_pdf(file: UploadFile = File(...)):
#     # Ensure file is a PDF
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="File must be a PDF")
    
#     try:
#         # Reset the file pointer to the start
#         file_content = await file.read()
#         file.file.seek(0)  # Reset the file pointer
        
#         # Extract content from the PDF
#         content = ""
#         with fitz.open(stream=file_content, filetype="pdf") as doc:
#             for page in doc:
#                 content += page.get_text()

#         # Save to the database
#         db = SessionLocal()
#         db_document = Document(filename=file.filename, content=content)
#         db.add(db_document)
#         db.commit()
#         db.refresh(db_document)
#         return {"document_id": db_document.id}
    
#     except Exception as e:
#         # Handle unexpected errors and close the database session
#         raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
#     finally:
#         # Ensure that the database session is closed even if an error occurs
#         db.close()
# import os
# import logging
# from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
# from sqlalchemy.orm import Session
# import fitz
# from app.models import Document
# from app.db import get_db

# router = APIRouter()
# logger = logging.getLogger(__name__)

# UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @router.post("/upload/")
# async def upload_pdf(
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are allowed")

#     try:
#         # Read file content
#         file_content = await file.read()
        
#         # Save file to disk
#         file_path = os.path.join(UPLOAD_DIR, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(file_content)

#         # Extract text content
#         content = ""
#         try:
#             with fitz.open(file_path) as doc:
#                 for page in doc:
#                     content += page.get_text()
#         except Exception as e:
#             logger.error(f"PDF parsing error: {e}")
#             os.remove(file_path)  # Clean up file if parsing fails
#             raise HTTPException(status_code=400, detail="Could not parse PDF file")

#         # Save to database
#         document = Document(
#             filename=file.filename,
#             file_path=file_path,
#             content=content
#         )
#         db.add(document)
#         db.commit()
#         db.refresh(document)

#         return {"document_id": document.id, "message": "File uploaded successfully"}

#     except Exception as e:
#         logger.error(f"Upload error: {e}")
#         raise HTTPException(status_code=500, detail="Error uploading file")



# import os
# import logging
# from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
# from sqlalchemy.orm import Session
# import fitz
# from app.models import Document
# from app.db import get_db
# import shutil

# router = APIRouter()
# logger = logging.getLogger(__name__)

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

# UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# def validate_pdf(file_path: str) -> bool:
#     try:
#         with fitz.open(file_path) as doc:
#             # Just checking if we can open it as a PDF
#             return doc.page_count > 0
#     except Exception as e:
#         logger.error(f"PDF validation failed: {str(e)}")
#         return False

# @router.post("/upload/")
# async def upload_pdf(
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     logger.info(f"Starting upload process for file: {file.filename}")
    
#     if not file.filename.endswith('.pdf'):
#         logger.warning(f"Invalid file type attempted: {file.filename}")
#         raise HTTPException(
#             status_code=400,
#             detail="Only PDF files are allowed"
#         )

#     try:
#         # Create a temporary file path
#         temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
#         final_file_path = os.path.join(UPLOAD_DIR, file.filename)

#         # Save uploaded file to temporary location
#         logger.debug(f"Saving file to temporary location: {temp_file_path}")
#         with open(temp_file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Validate PDF
#         if not validate_pdf(temp_file_path):
#             logger.error(f"Invalid or corrupted PDF file: {file.filename}")
#             os.remove(temp_file_path)
#             raise HTTPException(
#                 status_code=400,
#                 detail="Invalid or corrupted PDF file"
#             )

#         # Extract text content
#         logger.debug("Extracting text content from PDF")
#         content = ""
#         try:
#             with fitz.open(temp_file_path) as doc:
#                 for page in doc:
#                     content += page.get_text()
#         except Exception as e:
#             logger.error(f"PDF parsing error: {str(e)}")
#             os.remove(temp_file_path)
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Could not parse PDF file: {str(e)}"
#             )

#         # Move to final location
#         logger.debug(f"Moving file to final location: {final_file_path}")
#         shutil.move(temp_file_path, final_file_path)

#         # Save to database
#         logger.debug("Saving document information to database")
#         document = Document(
#             filename=file.filename,
#             file_path=final_file_path,
#             content=content
#         )
#         db.add(document)
#         db.commit()
#         db.refresh(document)

#         logger.info(f"Successfully processed file: {file.filename}")
#         return {
#             "document_id": document.id,
#             "message": "File uploaded successfully",
#             "filename": file.filename
#         }

#     except Exception as e:
#         logger.error(f"Upload error: {str(e)}", exc_info=True)
#         # Cleanup any temporary files
#         if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
#             os.remove(temp_file_path)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error uploading file: {str(e)}"
#         )