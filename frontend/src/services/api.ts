import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

// export const uploadPDF = async (file: File) => {
//   // Simulated successful response
//   return { document_id: 6 };
// };


export const uploadPDF = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await axios.post(`${API_BASE_URL}/upload/`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      validateStatus: (status) => status < 500,
    });

    console.log(response.data)

    if (response.status !== 200) {
      throw new Error(response.data.detail || 'Upload failed');
    }

    if (!response.data.document_id) {
      throw new Error('Invalid response from server');
    }

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message;
      console.error('Upload error:', errorMessage);
      throw new Error(`Failed to upload PDF: ${errorMessage}`);
    }
    throw error;
  }
};

export const askQuestion = async (documentId: number, question: string) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/question/`, {
      document_id: documentId,
      question: question
    });

    if (!response.data.answer) {
      throw new Error('Invalid response from server');
    }

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message;
      console.error('Question error:', errorMessage);
      throw new Error(`Failed to get answer: ${errorMessage}`);
    }
    throw error;
  }
};