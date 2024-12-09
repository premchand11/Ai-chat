# Full-Stack PDF Question-Answering Application

## Objective
The goal of this project is to develop a full-stack application that allows users to upload PDF documents and ask questions about the content of those documents. The backend processes the PDF documents, uses natural language processing (NLP), and provides answers to the questions posed by the users.

## Tools and Technologies
- **Backend:** FastAPI
- **NLP Processing:** LangChain / LlamaIndex
- **Frontend:** React.js
- **Database:** SQLite or PostgreSQL (to store document metadata, if necessary)
- **File Storage:** Local filesystem or cloud storage (e.g., AWS S3) for storing uploaded PDFs

## Functional Requirements
1. **PDF Upload:**
  - Users can upload PDF documents to the application.
  - The application stores the PDF and, if necessary, extracts and stores the text content for further processing.
2. **Asking Questions:**
  - Users can ask questions related to the content of an uploaded PDF.
  - The system processes the question along with the content of the PDF to provide an answer.
3. **Displaying Answers:**
  - The application displays the answer to the user’s question.
  - Users can ask follow-up or new questions on the same document.


## Backend Specification
### FastAPI Endpoints
- **Upload Endpoint:**
  - Allows users to upload PDF documents.
  - The endpoint stores the document and extracts its text for processing.
- **Question Endpoint:**
  - Accepts questions from users.
  - Uses the LangChain/llama index to process the question and return answers based on the PDF content.

### PDF Processing
- Extract text from uploaded PDFs using PyMuPDF or a similar library.
- Process natural language questions with LangChain/llama index to generate answers from the PDF content.

### Data Management
- Store document metadata (e.g., filename, upload date) in the database (SQLite or PostgreSQL).

## Frontend Specification
### User Interface
- **Upload Page:**
  - Allows users to upload PDF documents.
- **Questioning Interface:**
  - Users can ask questions related to the content of the uploaded document.
  - Display answers in a user-friendly manner.

### Interactivity
- Provided feedback mechanisms for document uploads and question processing.
- Displayed error messages for unsupported file types or processing errors.

## Setup Instructions
### Prerequisites
- Python 3.x
- Node.js and npm
- FastAPI
- React.js
- SQLite/PostgreSQL (optional)

### Backend Setup
1. Clone the repository:
  ```bash
  git clone <repository_url>
  ```
2. Install Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Start the FastAPI server:
  ```bash
  uvicorn app.main:app --reload
  ```

### Frontend Setup
1. Navigate to the frontend directory:
  ```bash
  cd frontend
  ```
2. Install React dependencies:
  ```bash
  npm install
  ```
3. Start the React development server:
  ```bash
  npm run dev
  ```

### Database Setup
1. Create a database (SQLite/PostgreSQL) and configure it for storing document metadata (optional).
2. Update the `database.py` file with the appropriate database credentials and configuration.

### PDF Processing
1. Ensure that you have installed the required libraries for PDF text extraction (e.g., PyMuPDF).
2. Configure LangChain/llama index for processing questions.



## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- LangChain for the NLP question-answering framework.
- FastAPI for building the backend API.
- React.js for the frontend development.
- PyMuPDF for PDF text extraction.

## Video Demonstration
[Watch the video demonstration here](<https://drive.google.com/file/d/1NTN1t6efNgx7qEdYQUvmeSlvb2qtM_Fx/view?usp=drive_link>)
