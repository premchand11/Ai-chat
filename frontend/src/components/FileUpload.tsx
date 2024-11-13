import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileUp, File, Loader2 } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isLoading }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    disabled: isLoading
  });

  return (
    <div
      {...getRootProps()}
      className={`w-full h-[600px] flex items-center justify-center border-2 border-dashed rounded-xl transition-colors duration-200 ease-in-out ${
        isLoading ? 'cursor-not-allowed opacity-75' : 'cursor-pointer'
      } ${
        isDragActive 
          ? 'border-[#6ef792] bg-[#F9F5FF]' 
          : 'border-gray-200 hover:border-[#72f57f] hover:bg-[#F9F5FF]'
      }`}
    >
      <input {...getInputProps()} id="fileInput" />
      <div className="flex flex-col items-center justify-center text-center">
        <div className="mb-4">
          {isLoading ? (
            <Loader2 className="w-12 h-12 text-[#8ef173] animate-spin" />
          ) : isDragActive ? (
            <File className="w-12 h-12 text-[#72fb89]" />
          ) : (
            <FileUp className="w-12 h-12 text-gray-400" />
          )}
        </div>
        <p className="text-lg font-medium mb-2 text-gray-700">
          {isLoading 
            ? 'Uploading...' 
            : isDragActive 
              ? 'Drop your PDF here' 
              : 'Upload your PDF'}
        </p>
        {!isLoading && (
          <p className="text-sm text-gray-500">
            or click to browse from your computer
          </p>
        )}
      </div>
    </div>
  );
};