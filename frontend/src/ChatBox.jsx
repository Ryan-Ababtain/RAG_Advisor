import React, { useState } from 'react';
import axios from 'axios';

export default function ChatBox() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [model, setModel] = useState('llama3');

  const handleAsk = async () => {
    const res = await axios.get('/ask', { params: { query, model } });
    setAnswer(res.data.answer);
  };

  return (
    <div>
      <select value={model} onChange={e => setModel(e.target.value)}>
        <option value="llama3">llama3</option>
        <option value="mistral">mistral</option>
      </select>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      <button onClick={handleAsk}>Ask</button>
      <div>{answer}</div>
    </div>
  );
}
