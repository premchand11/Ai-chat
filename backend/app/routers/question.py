import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.models import Document
from app.db import get_db
from typing import Dict, Any
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str
    document_id: int

# Custom prompt template
QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant. Based on the following context, answer the question below as accurately as possible. If the answer is not found in the context, say "I cannot find the answer in the provided document."

Context: {context}

Question: {question}

Answer: """,
    input_variables=["context", "question"]
)

def create_vector_store(content: str):
    try:
        # Local embeddings model for context representation
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Text splitter with refined chunking for better document context handling
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", ",", " "]
        )
        
        texts = text_splitter.split_text(content)
        logger.info(f"Split content into {len(texts)} chunks")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_texts(
            texts, 
            embeddings_model,
            metadatas=[{"source": f"chunk_{i}"} for i in range(len(texts))]
        )
        
        return vectorstore, texts
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting up the QA system.")

def retrieve_answer(vectorstore, question: str) -> str:
    try:
        # Stub or placeholder for LLM; substitute with local model if available
        llm = None  # Replace with a local LLM loading code if available
        
        # Log and return an error if no local LLM is available
        if llm is None:
            logger.error("Local LLM model not configured.")
            return "Local LLM model not configured. Please set up a local model."
        
        # Set up the retrieval chain with the prompt template
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 15,
                    "score_threshold": 0.3
                }
            ),
            chain_type_kwargs={
                "prompt": QA_PROMPT,
                "verbose": True
            },
            return_source_documents=True
        )
        
        # Get the answer
        result = qa_chain({"query": question})
        
        # Ensure the result is properly formatted
        answer = result.get("result", "No answer found").strip()
        
        return answer
    
    except Exception as e:
        logger.error(f"Error retrieving answer: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating answer.")

@router.post("/question/")
async def answer_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    try:
        # Retrieve the document from the database
        document = db.query(Document).filter(Document.id == request.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.content:
            raise HTTPException(status_code=400, detail="Document has no content")

        # Process the question and get the answer
        logger.info(f"Processing question for document {request.document_id}: {request.question}")
        
        # Create the vector store and retrieve the answer
        vectorstore, texts = create_vector_store(document.content)
        answer = retrieve_answer(vectorstore, request.question)
        
        return {
            "answer": answer,
            "status": "success",
            "document_id": request.document_id
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
