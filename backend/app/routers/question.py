
# import os
# import logging
# from fastapi import APIRouter, HTTPException, Depends, Request
# from sqlalchemy.orm import Session
# from app.models import Document
# from app.db import get_db
# from typing import Dict, Any
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# class QuestionRequest(BaseModel):
#     question: str
#     document_id: int # changed herrrerereer

# def get_local_embeddings_model():
#     # Initialize Sentence Transformers with a local model
#     model_name = "all-mpnet-base-v2"
#     return SentenceTransformer(model_name)

# def get_faiss_index(content: str):
#     try:
#         # Initialize the embeddings model
#         embeddings_model = get_local_embeddings_model()
        
#         # Split content into chunks to avoid large input size
#         max_chunk_size = 600
#         texts = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
        
#         # Generate embeddings for each chunk
#         embeddings = embeddings_model.encode(texts, convert_to_tensor=False)  # Directly to numpy array
        
#         # Initialize a FAISS index and add embeddings
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
#         index.add(np.array(embeddings))

#         # Return the index and text chunks for retrieval
#         return index, texts, embeddings_model
#     except Exception as e:
#         logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error setting up the QA system: {str(e)}"
#         )

# def retrieve_answer(index, texts, question, embeddings_model):
#     # Convert question to embedding and retrieve similar chunks from FAISS
#     question_embedding = embeddings_model.encode([question], convert_to_tensor=False)
#     _, indices = index.search(np.array(question_embedding), k=2)  # Top 3 closest chunks
    
#     # Concatenate retrieved chunks as the "answer"
#     answer = " ".join([texts[i] for i in indices[0]])
#     return answer

# @router.post("/question/")
# async def answer_question(
#     # document_id: int,
#     # question: str,
#     request: QuestionRequest,
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     document_id = request.document_id
#     question = request.question
#     logger.info(f"Processing question for document ID: {document_id}")
    
    
#     try:
#         # Get document from database
#         document = db.query(Document).filter(Document.id == document_id).first()
#         if not document:
#             logger.warning(f"Document not found: {document_id}")
#             raise HTTPException(
#                 status_code=404,
#                 detail="Document not found"
#             )
        
#         if not document.content:
#             logger.error(f"Document {document_id} has no content")
#             raise HTTPException(
#                 status_code=400,
#                 detail="Document has no content to analyze"
#             )

#         # Create FAISS index and retrieve answer
#         index, texts, embeddings_model = get_faiss_index(document.content)
#         answer = retrieve_answer(index, texts, question, embeddings_model)
        
#         if not answer:
#             logger.warning("No answer generated")
#             return {
#                 "answer": "I couldn't find a specific answer to your question in the document."
#             }
        
#         logger.info("Successfully generated answer")
#         return {"answer": answer}

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing question: {str(e)}"
#         )

# previousss
   
# import os
# import logging
# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.models import Document
# from app.db import get_db
# from typing import Dict, Any
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# class QuestionRequest(BaseModel):
#     question: str
#     document_id: int

# def get_local_embeddings_model():
#     model_name = "paraphrase-MiniLM-L6-v2"
#     return SentenceTransformer(model_name)

# def get_faiss_index(content: str):
#     try:
#         embeddings_model = get_local_embeddings_model()
        
#         # Split content into smaller chunks
#         max_chunk_size = 1000  # Try a smaller chunk size
#         texts = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
        
#         # Generate embeddings and normalize for cosine similarity
#         embeddings = embeddings_model.encode(texts, convert_to_tensor=False)
#         embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

#         # Initialize a cosine similarity index with FAISS
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
#         index.add(np.array(embeddings))

#         return index, texts, embeddings_model
#     except Exception as e:
#         logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error setting up the QA system: {str(e)}")

# def retrieve_answer(index, texts, question, embeddings_model):
#     # Normalize question embedding for cosine similarity
#     question_embedding = embeddings_model.encode([question], convert_to_tensor=False)
#     question_embedding = question_embedding / np.linalg.norm(question_embedding)

#     # Retrieve top 2 chunks for conciseness
#     _, indices = index.search(np.array(question_embedding), k=2)
#     answer = " ".join([texts[i] for i in indices[0]])

#     return answer

# @router.post("/question/")
# async def answer_question(
#     request: QuestionRequest,
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     document_id = request.document_id
#     question = request.question
#     logger.info(f"Processing question for document ID: {document_id}")
    
#     try:
#         document = db.query(Document).filter(Document.id == document_id).first()
#         if not document:
#             logger.warning(f"Document not found: {document_id}")
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         if not document.content:
#             logger.error(f"Document {document_id} has no content")
#             raise HTTPException(status_code=400, detail="Document has no content to analyze")

#         # Create FAISS index and retrieve answer
#         index, texts, embeddings_model = get_faiss_index(document.content)
#         answer = retrieve_answer(index, texts, question, embeddings_model)
        
#         if not answer:
#             logger.warning("No answer generated")
#             return {"answer": "I couldn't find a specific answer to your question in the document."}
        
#         logger.info("Successfully generated answer")
#         return {"answer": answer}

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# model 1
# import os
# import logging
# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.models import Document
# from app.db import get_db
# from typing import Dict, Any
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import pipeline

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# class QuestionRequest(BaseModel):
#     question: str
#     document_id: int

# def get_local_embeddings_model():
#     model_name = "paraphrase-MiniLM-L6-v2"
#     return SentenceTransformer(model_name)

# def get_faiss_index(content: str):
#     try:
#         embeddings_model = get_local_embeddings_model()
        
#         # Split content into smaller chunks
#         max_chunk_size = 3000  # Try a smaller chunk size
#         texts = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
        
#         # Generate embeddings
#         embeddings = embeddings_model.encode(texts, convert_to_tensor=False)
        
#         # Ensure embeddings is a 2D array
#         if embeddings.ndim == 1:
#             embeddings = embeddings.reshape(1, -1)  # Reshape to 2D if it's 1D
        
#         # Normalize embeddings for cosine similarity
#         embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

#         # Initialize a cosine similarity index with FAISS
#         dimension = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
#         index.add(np.array(embeddings))

#         return index, texts, embeddings_model
#     except Exception as e:
#         logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error setting up the QA system: {str(e)}")
    

# def generate_answer(chunks, question):
#     # Use a pre-trained QA model (e.g., DistilBERT or T5)
#     qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
#     context = " ".join(chunks)
#     answer = qa_pipeline(question=question, context=context)
#     return answer['answer']

# def retrieve_answer(index, texts, question, embeddings_model):
#     # Normalize question embedding for cosine similarity
#     question_embedding = embeddings_model.encode([question], convert_to_tensor=False)
#     question_embedding = question_embedding / np.linalg.norm(question_embedding)

#     # Retrieve top 5 chunks for conciseness
#     _, indices = index.search(np.array(question_embedding), k=5)  # Increased k for better context
#     top_chunks = [texts[i] for i in indices[0]]

#     # Use a QA model to generate a more coherent answer from the chunks
#     return generate_answer(top_chunks, question)

# @router.post("/question/")
# async def answer_question(
#     request: QuestionRequest,
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     document_id = request.document_id
#     question = request.question
#     logger.info(f"Processing question for document ID: {document_id}")
    
#     try:
#         # Retrieve document from DB
#         document = db.query(Document).filter(Document.id == document_id).first()
#         if not document:
#             logger.warning(f"Document not found: {document_id}")
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         if not document.content:
#             logger.error(f"Document {document_id} has no content")
#             raise HTTPException(status_code=400, detail="Document has no content to analyze")

#         # Create FAISS index and retrieve answer
#         index, texts, embeddings_model = get_faiss_index(document.content)
#         answer = retrieve_answer(index, texts, question, embeddings_model)
        
#         if not answer:
#             logger.warning("No answer generated")
#             return {"answer": "I couldn't find a specific answer to your question in the document."}
        
#         logger.info("Successfully generated answer")
#         return {"answer": answer}

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

#model 2

# import os
# import logging
# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.models import Document
# from app.db import get_db
# from typing import Dict, Any
# from pydantic import BaseModel
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
# from langchain_community.vectorstores import FAISS  # Updated import
# from transformers import pipeline  # Import HuggingFace's pipeline  # Keep prompt template as it is
# from langchain_community.chains import ConversationalRetrievalChain  # Correct chain import
# # from langchain_community.llms import HuggingFaceLLM  # Updated import
# from langchain.prompts import PromptTemplate  # Prompt remains unchanged
# from langchain_community.llms import HuggingFacePipeline  # Updated import



# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# class QuestionRequest(BaseModel):
#     question: str
#     document_id: int

# def get_embeddings_model():
#     """Return the HuggingFace Embedding model (RoBERTa or similar)."""
#     model_name = "roberta-base"  # You can switch to any model of your choice, e.g., "t5-small"
#     return HuggingFaceEmbeddings(model_name=model_name)

# def get_faiss_index(content: str, embedding_model) -> FAISS:
#     """Create FAISS index using document content and embedding model."""
#     try:
#         # Split content into smaller chunks (to avoid large input tokens for models)
#         max_chunk_size = 3000  # Adjust chunk size based on document length
#         texts = [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]
        
#         # Generate embeddings for the text chunks
#         embeddings = embedding_model.encode(texts)
        
#         # Initialize FAISS index for cosine similarity search
#         faiss_index = FAISS.from_documents(texts, embedding_model)
        
#         return faiss_index, texts
#     except Exception as e:
#         logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error setting up the QA system: {str(e)}")

# # def get_qa_chain(faiss_index: FAISS) -> ConversationalRetrievalChain:
# #     """Set up a question-answering chain using T5 or any HuggingFace LLM."""
# #     # Initialize generative model (T5 or any other desired model)
# #     t5_model = HuggingFaceLLM(model_name="t5-small")  # You can also use other models like GPT-3, T5-large, etc.
    
# #     # Set up the conversational retrieval chain using FAISS and the generative model
# #     qa_chain = ConversationalRetrievalChain.from_llm_and_retriever(
# #         llm=t5_model, retriever=faiss_index.as_retriever()
# #     )
    
# #     return qa_chain

# def get_qa_chain(faiss_index: FAISS) -> ConversationalRetrievalChain:
#     """Set up a question-answering chain using HuggingFace's pipeline."""
    
#     # Initialize the HuggingFace model using the pipeline
#     pipe = pipeline("text2text-generation", model="t5-small")  # Use any HuggingFace model, e.g., T5
    
#     # Create HuggingFacePipeline
#     hf_pipeline = HuggingFacePipeline(pipeline=pipe)
    
#     # Set up the conversational retrieval chain using FAISS and the HuggingFace model
#     qa_chain = ConversationalRetrievalChain.from_llm_and_retriever(
#         llm=hf_pipeline, retriever=faiss_index.as_retriever()
#     )
    
#     return qa_chain

# @router.post("/question/")
# async def answer_question(
#     request: QuestionRequest,
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     document_id = request.document_id
#     question = request.question
#     logger.info(f"Processing question for document ID: {document_id}")
    
#     try:
#         # Retrieve document from DB
#         document = db.query(Document).filter(Document.id == document_id).first()
#         if not document:
#             logger.warning(f"Document not found: {document_id}")
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         if not document.content:
#             logger.error(f"Document {document_id} has no content")
#             raise HTTPException(status_code=400, detail="Document has no content to analyze")
        
#         # Retrieve the embedding model and create FAISS index
#         embedding_model = get_embeddings_model()
#         faiss_index, texts = get_faiss_index(document.content, embedding_model)
        
#         # Set up the QA chain
#         qa_chain = get_qa_chain(faiss_index)
        
#         # Get the answer using the question and the retrieval-based QA system
#         answer = qa_chain.run(question)
        
#         if not answer:
#             logger.warning("No answer generated")
#             return {"answer": "I couldn't find a specific answer to your question in the document."}
        
#         logger.info("Successfully generated answer")
#         return {"answer": answer}

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
 #model 3

# import os
# import logging
# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.models import Document
# from app.db import get_db
# from typing import Dict, Any
# from pydantic import BaseModel
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import LlamaIndex
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFacePipeline
# from llama_index import GPTVectorStoreIndex
# from llama_index.core.indices import VectorStoreIndex
# from llama_index.core.query_engine import BaseQueryEngine

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# class QuestionRequest(BaseModel):
#     question: str
#     document_id: int

# def get_llama_index(content: str):
#     try:
#         embeddings_model = SentenceTransformerEmbeddings("paraphrase-MiniLM-L6-v2")
        
#         # Split content into smaller chunks
#         max_chunk_size = 3000  # Chunk size for LlamaIndex
#         texts = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
        
#         # Create LlamaIndex from document chunks and embeddings
#         index = LlamaIndex.from_texts(texts, embeddings_model)
        
#         return index, texts
#     except Exception as e:
#         logger.error(f"Error creating LlamaIndex: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error setting up the QA system: {str(e)}")
    
# def retrieve_answer(index, question):
#     # Configure the RetrievalQA chain to answer questions
#     qa_chain = RetrievalQA.from_chain_type(llm=HuggingFacePipeline("distilbert-base-uncased-distilled-squad"), retriever=index.as_retriever())
    
#     # Generate the answer
#     answer = qa_chain.run(question)
#     return answer or "I couldn't find a specific answer to your question in the document."

# @router.post("/question/")
# async def answer_question(
#     request: QuestionRequest,
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     document_id = request.document_id
#     question = request.question
#     logger.info(f"Processing question for document ID: {document_id}")
    
#     try:
#         # Retrieve document from DB
#         document = db.query(Document).filter(Document.id == document_id).first()
#         if not document:
#             logger.warning(f"Document not found: {document_id}")
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         if not document.content:
#             logger.error(f"Document {document_id} has no content")
#             raise HTTPException(status_code=400, detail="Document has no content to analyze")

#         # Create LlamaIndex and retrieve answer
#         index, texts = get_llama_index(document.content)
#         answer = retrieve_answer(index, question)
        
#         logger.info("Successfully generated answer")
#         return {"answer": answer}

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
  #model fin

# import os
# import logging
# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.models import Document
# from app.db import get_db
# from typing import Dict, Any
# from pydantic import BaseModel

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Enhanced logging configuration
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# router = APIRouter()

# class QuestionRequest(BaseModel):
#     question: str
#     document_id: int

# def get_llm():
#     """Initialize a local LLM using HuggingFace's pipeline"""
#     try:
#         logger.info("Initializing language model...")

#         model_id = "google/flan-t5-base"
#         pipe = pipeline(
#             "text2text-generation",
#             model=model_id,
#             max_length=512,
#             temperature=0.3,
#             top_p=0.95,
#             repetition_penalty=1.15
#         )

#         local_llm = HuggingFacePipeline(pipeline=pipe)
#         logger.info(f"Successfully initialized language model: {model_id}")
#         return local_llm
    
#     except Exception as e:
#         logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error initializing language model: {str(e)}"
#         )

# def get_document_store(content: str):
#     try:
#         logger.info("Creating document store...")

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=100,
#             length_function=len,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         texts = text_splitter.create_documents([content])
#         logger.debug(f"Document split into {len(texts)} chunks")

#         logger.info("Initializing embeddings model...")
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2"
#         )

#         logger.info("Creating FAISS index...")
#         vectorstore = FAISS.from_documents(
#             documents=texts,
#             embedding=embeddings
#         )
        
#         logger.debug(f"FAISS index created with {len(texts)} chunks")
#         return vectorstore, texts
    
#     except Exception as e:
#         logger.error(f"Error creating document store: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Error setting up the QA system: {str(e)}"
#         )

# def get_qa_chain(vectorstore):
#     try:
#         logger.info("Creating QA chain...")

#         retriever = vectorstore.as_retriever(
#             search_type="mmr",
#             search_kwargs={
#                 "k": 5,
#                 "fetch_k": 8
#             }
#         )
#         local_llm = get_llm()

#         template = """Given the following context and question, provide a detailed and accurate answer. If the answer cannot be found in the context, say "I cannot find an answer to this question in the document."

# Context: {context}

# Question: {question}

# Provide a comprehensive answer based on the context above:"""
        
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=local_llm,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True,
#             chain_type_kwargs={
#                 "verbose": True,
#                 "prompt": template
#             }
#         )
        
#         logger.info("QA chain successfully created")
#         return qa_chain
    
#     except Exception as e:
#         logger.error(f"Error creating QA chain: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error setting up QA chain: {str(e)}"
#         )

# @router.post("/question/")
# async def answer_question(
#     request: QuestionRequest,
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     document_id = request.document_id
#     question = request.question
#     logger.info(f"Processing question: '{question}' for document ID: {document_id}")

#     try:
#         document = db.query(Document).filter(Document.id == document_id).first()
#         if not document:
#             logger.warning(f"Document not found: {document_id}")
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         if not document.content:
#             logger.error(f"Document {document_id} has no content")
#             raise HTTPException(
#                 status_code=400,
#                 detail="Document has no content to analyze"
#             )

#         logger.info("Processing document content...")
#         vectorstore, texts = get_document_store(document.content)
#         qa_chain = get_qa_chain(vectorstore)
        
#         logger.debug(f"Document chunks created: {len(texts)}")
#         for i, text in enumerate(texts[:2]):
#             logger.debug(f"Chunk {i} preview: {text.page_content[:100]}...")
        
#         logger.info("Generating answer...")
#         result = qa_chain({"query": question})
        
#         logger.info("Retrieved source chunks:")
#         for i, doc in enumerate(result["source_documents"]):
#             logger.debug(f"Source {i+1}: {doc.page_content[:200]}...")
        
#         if not result["result"]:
#             logger.warning("No answer generated")
#             return {
#                 "answer": "I couldn't find a specific answer to your question in the document."
#             }
        
#         answer_response = {
#             "answer": result["result"],
#             "sources": [
#                 {
#                     "content": doc.page_content[:200] + "...",
#                     "metadata": doc.metadata
#                 }
#                 for doc in result["source_documents"]
#             ],
#             "metadata": {
#                 "chunks_used": len(result["source_documents"]),
#                 "total_chunks": len(texts)
#             }
#         }
        
#         logger.info(f"Generated answer: {result['result'][:200]}...")
#         return answer_response

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing question: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing question: {str(e)}"
#         )
# import os
# import logging
# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.models import Document
# from app.db import get_db
# from typing import Dict, Any
# from pydantic import BaseModel
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFaceHub
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# # Set HuggingFace API token
# os.environ["HUGGINGFACE_API_TOKEN"] = "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"
# HUGGINGFACE_API_TOKEN = os.environ["HUGGINGFACE_API_TOKEN"]

# class QuestionRequest(BaseModel):
#     question: str
#     document_id: int

# # Custom prompt template
# QA_PROMPT = PromptTemplate(
#     template="""You are a helpful assistant. Based on the following context, answer the question below as accurately as possible. If the answer is not found in the context, say "I cannot find the answer in the provided document."

# Context: {context}

# Question: {question}

# Answer: """,
#     input_variables=["context", "question"]
# )

# def create_vector_store(content: str):
#     try:
#         # Using FLAN-T5 specific embeddings model for better context representation
#         embeddings_model = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/paraphrase-mpnet-base-v2",  # Better embeddings for QA tasks
#             model_kwargs={'device': 'cpu'}
#         )
        
#         # Text splitter with refined chunking for better document context handling
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,  # Optimal chunk size for better retrieval
#             chunk_overlap=50,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", ",", " "]
#         )
        
#         texts = text_splitter.split_text(content)
#         logger.info(f"Split content into {len(texts)} chunks")
        
#         # Create FAISS vector store
#         vectorstore = FAISS.from_texts(
#             texts, 
#             embeddings_model,
#             metadatas=[{"source": f"chunk_{i}"} for i in range(len(texts))]
#         )
        
#         return vectorstore, texts
#     except Exception as e:
#         logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error setting up the QA system: {str(e)}")

# def retrieve_answer(vectorstore, question: str) -> str:
#     try:
#         # Use FLAN-T5 (flan-t5-xl) for better comprehension and more accurate answers
#         llm = HuggingFaceHub(
#             repo_id="google/flan-t5-xl",  # FLAN-T5 model for question answering
#             huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
#             model_kwargs={
#                 "temperature": 0.2,  # Stable output, focused answers
#                 "max_length": 1024,
#                 "top_p": 0.9,
#                 "num_return_sequences": 1
#             }
#         )
        
#         # Set up the retrieval chain
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={
#                     "k": 15,  # Retrieve 5 relevant chunks for better accuracy
#                     "score_threshold": 0.3
#                 }
#             ),
#             chain_type_kwargs={
#                 "prompt": QA_PROMPT,
#                 "verbose": True
#             },
#             return_source_documents=True
#         )
        
#         # Get the answer
#         result = qa_chain({"query": question})
        
#         # Ensure the result is properly formatted
#         answer = result.get("result", "No answer found").strip()
        
#         return answer
    
#     except Exception as e:
#         logger.error(f"Error retrieving answer: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

# @router.post("/question/")
# async def answer_question(
#     request: QuestionRequest,
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     try:
#         # Retrieve the document from the database
#         document = db.query(Document).filter(Document.id == request.document_id).first()
#         if not document:
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         if not document.content:
#             raise HTTPException(status_code=400, detail="Document has no content")

#         # Process the question and get the answer
#         logger.info(f"Processing question for document {request.document_id}: {request.question}")
        
#         # Create the vector store and retrieve the answer
#         vectorstore, texts = create_vector_store(document.content)
#         answer = retrieve_answer(vectorstore, request.question)
        
#         return {
#             "answer": answer,
#             "status": "success",
#             "document_id": request.document_id
#         }

#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         logger.error(f"Error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

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
