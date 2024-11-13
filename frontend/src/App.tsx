import React, { useState } from 'react';
import { QuestionInput } from './components/QuestionInput';
import { ChatMessage } from './components/ChatMessage';
import { FileText, Plus } from 'lucide-react';
import { FileUpload } from './components/FileUpload';
import { uploadPDF, askQuestion } from './services/api';

interface Message {
  text: string;
  isUser: boolean;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentId, setDocumentId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await uploadPDF(file);
      setSelectedFile(file);
      setDocumentId(response.document_id);
      setMessages([]);
    } catch (err) {
      setError('Failed to upload PDF. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAskQuestion = async (question: string) => {
    if (!documentId) return;

    setMessages(prev => [...prev, { text: question, isUser: true }]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await askQuestion(documentId, question);
      setMessages(prev => [...prev, {
        text: response.answer,
        isUser: false
      }]);
    } catch (err) {
      setError('Failed to get answer. Please try again.');
      console.error('Question error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <div className="w-full border-b border-gray-100">
        <div className="max-w-[1200px] mx-auto px-6">
          <header className="flex items-center justify-between py-4">
            <img src="https://framerusercontent.com/images/pFpeWgK03UT38AQl5d988Epcsc.svg?scale-down-to=512" alt="AI Planet" className="h-8" />
            <div className="flex items-center gap-2">
              {selectedFile && (
                <>
                  <FileText className="w-5 h-5 text-green-500" />
                  <span className="text-sm text-gray-600">{selectedFile.name}</span>
                </>
              )}
              <label className="flex items-center gap-2 px-4 py-2 rounded-full border border-gray-200 hover:border-gray-300 transition-colors cursor-pointer ml-4">
                <Plus className="w-5 h-5" />
                <span>Upload PDF</span>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                  className="hidden"
                  disabled={isLoading}
                />
              </label>
            </div>
          </header>
        </div>
      </div>

      <div className="flex-1 max-w-[1200px] mx-auto px-6 w-full">
        {error && (
          <div className="my-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-600">
            {error}
          </div>
        )}

        {!selectedFile && !isLoading && (
          <FileUpload onFileSelect={handleFileUpload} isLoading={isLoading} />
        )}

        {messages.length > 0 && (
          <div className="flex-1 overflow-y-auto space-y-6 py-6">
            {messages.map((message, index) => (
              <ChatMessage
                key={index}
                message={message.text}
                isUser={message.isUser}
              />
            ))}
            {isLoading && (
              <div className="flex items-center justify-center py-4">
                <div className="animate-pulse text-gray-400">Processing...</div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="border-t border-gray-100">
        <div className="max-w-[1200px] mx-auto px-6 py-6">
          <QuestionInput
            onAsk={handleAskQuestion}
            disabled={isLoading || !selectedFile}
            placeholder={selectedFile ? "Send a message..." : "Upload a PDF to start chatting"}
          />
        </div>
      </div>
    </div>
  );
}

export default App;