require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { QdrantClient } = require('@qdrant/js-client-rest');

const app = express();
app.use(bodyParser.json());

const config = {
  instruction: '',
  temperature: 0.7,
  topP: 1,
  topK: 3,
};

const qdrant = new QdrantClient({ url: process.env.QDRANT_URL || 'http://localhost:6333' });
const collection = 'docs';

async function ensureCollection() {
  try {
    await qdrant.createCollection(collection, { vectors: { size: 1536, distance: 'Cosine' } });
  } catch (e) {
    // Collection probably exists
  }
}

async function ingestDocument(text) {
  const embeddings = new OpenAIEmbeddings({ apiKey: process.env.OPENAI_API_KEY });
  await ensureCollection();
  const vector = await embeddings.embedQuery(text);
  await qdrant.upsert(collection, {
    points: [{ id: Date.now(), vector, payload: { text } }],
  });
}

async function searchDocs(query) {
  const embeddings = new OpenAIEmbeddings({ apiKey: process.env.OPENAI_API_KEY });
  const vector = await embeddings.embedQuery(query);
  const results = await qdrant.search(collection, {
    vector,
    limit: config.topK,
    with_payload: true,
  });
  return results.map((r) => r.payload.text).join('\n');
}

async function askLLM(context, question) {
  const res = await axios.post(
    'https://api.openai.com/v1/chat/completions',
    {
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: config.instruction || 'You are a helpful assistant.' },
        {
          role: 'user',
          content: context ? `${context}\n\n${question}` : question,
        },
      ],
      temperature: config.temperature,
      top_p: config.topP,
    },
    { headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` } },
  );
  return res.data.choices[0].message.content.trim();
}

function adminHtml() {
  return `
    <h1>Admin</h1>
    <form id="upload-form">
      Instruction:<br/><input id="instruction" value="${config.instruction}"/><br/>
      Temperature:<br/><input id="temperature" type="number" step="0.1" value="${config.temperature}"/><br/>
      Top P:<br/><input id="topP" type="number" step="0.1" value="${config.topP}"/><br/>
      Top K:<br/><input id="topK" type="number" value="${config.topK}"/><br/>
      Document:<br/><input type="file" id="file" accept=".txt"/><br/><br/>
      <button type="submit">Upload</button>
    </form>
    <p id="status"></p>
    <a href="/chat">Go to Chat</a>
    <script>
      document.getElementById('upload-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const f = document.getElementById('file').files[0];
        const text = f ? await f.text() : '';
        const body = {
          instruction: document.getElementById('instruction').value,
          temperature: parseFloat(document.getElementById('temperature').value),
          topP: parseFloat(document.getElementById('topP').value),
          topK: parseInt(document.getElementById('topK').value, 10),
          text,
        };
        const res = await fetch('/admin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        document.getElementById('status').innerText = res.ok ? 'Uploaded!' : 'Upload failed';
      });
    </script>
  `;
}

function chatHtml() {
  return `
    <h1>Chatbot</h1>
    <div id="messages"></div>
    <input id="msg" placeholder="Ask something..."/>
    <button id="send">Send</button>
    <a href="/admin">Back to Admin</a>
    <script>
      document.getElementById('send').addEventListener('click', async () => {
        const msgEl = document.getElementById('msg');
        const msg = msgEl.value;
        if(!msg) return;
        msgEl.value = '';
        const chat = document.getElementById('messages');
        chat.innerHTML += '<p><b>You:</b> '+msg+'</p>';
        const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ message: msg }) });
        const data = await res.json();
        chat.innerHTML += '<p><b>Bot:</b> '+data.answer+'</p>';
      });
    </script>
  `;
}

function homeHtml() {
  return `
    <h1>RAG Chatbot Demo</h1>
    <p><a href="/admin">Admin</a></p>
    <p><a href="/chat">Chat</a></p>
  `;
}

app.get('/admin', (req, res) => {
  res.send(adminHtml());
});

app.post('/admin', async (req, res) => {
  const { instruction, temperature, topP, topK, text } = req.body;
  if (instruction !== undefined) config.instruction = instruction;
  if (!isNaN(temperature)) config.temperature = temperature;
  if (!isNaN(topP)) config.topP = topP;
  if (!isNaN(topK)) config.topK = topK;
  if (text) await ingestDocument(text);
  res.json({ status: 'ok' });
});

app.get('/chat', (req, res) => {
  res.send(chatHtml());
});

app.get('/', (req, res) => {
  res.send(homeHtml());
});

app.post('/chat', async (req, res) => {
  const { message } = req.body;
  try {
    const context = await searchDocs(message);
    const answer = await askLLM(context, message);
    res.json({ answer });
  } catch (e) {
    res.status(500).json({ error: 'Failed to generate answer' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Server running on port', PORT));
