import React from 'react';
import axios from 'axios';
import FileUploader from './FileUploader';
import ChatBox from './ChatBox';

export default function App() {
  const handleIngest = async () => {
    await axios.post('/ingest');
  };

  return (
    <div>
      <h1>RAG Advisor</h1>
      <FileUploader />
      <button onClick={handleIngest}>Ingest</button>
      <ChatBox />
    </div>
  );
}
