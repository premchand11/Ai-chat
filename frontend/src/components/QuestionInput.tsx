import React, { useState } from 'react';
import { Send } from 'lucide-react';

interface QuestionInputProps {
  onAsk: (question: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export const QuestionInput: React.FC<QuestionInputProps> = ({ 
  onAsk, 
  disabled,
  placeholder = "Send a message..." 
}) => {
  const [question, setQuestion] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim()) {
      onAsk(question);
      setQuestion('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className="w-full px-6 py-4 rounded-full bg-gray-50 border border-gray-200 focus:ring-2 focus:ring-[#7bf66e] focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed text-sm"
        />
        <button
          type="submit"
          disabled={disabled || !question.trim()}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-2 text-[#7F56D9] hover:text-[#41c64c] disabled:text-gray-400 disabled:cursor-not-allowed"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>
    </form>
  );
};