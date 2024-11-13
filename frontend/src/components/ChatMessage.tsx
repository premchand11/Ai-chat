import React from 'react';

interface ChatMessageProps {
  message: string;
  isUser: boolean;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, isUser }) => {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} gap-3`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-[rgb(108,236,129)] flex items-center justify-center text-white text-sm">
          AI
        </div>
      )}
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-[#E4E7EC] flex items-center justify-center text-gray-600 text-sm">
          S
        </div>
      )}
      <div
        className={`max-w-[80%] p-4 rounded-lg ${
          isUser
            ? 'bg-gray-100 text-gray-800'
            : 'bg-white border border-gray-200 text-gray-800'
        }`}
      >
        <p className="text-sm leading-relaxed">{message}</p>
      </div>
    </div>
  );
};