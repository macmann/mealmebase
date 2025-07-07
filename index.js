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

function pageTemplate(content) {
  return `
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css" />
        <title>RAG Chatbot</title>
      </head>
      <body>
        <section class="section">
          <div class="container">
            ${content}
          </div>
        </section>
      </body>
    </html>
  `;
}

function adminHtml() {
  return pageTemplate(`
    <h1 class="title">Admin</h1>
    <form id="upload-form">
      <div class="field">
        <label class="label">Instruction</label>
        <div class="control">
          <input class="input" id="instruction" value="${config.instruction}" />
        </div>
      </div>
      <div class="field">
        <label class="label">Temperature</label>
        <div class="control">
          <input class="input" id="temperature" type="number" step="0.1" value="${config.temperature}" />
        </div>
      </div>
      <div class="field">
        <label class="label">Top P</label>
        <div class="control">
          <input class="input" id="topP" type="number" step="0.1" value="${config.topP}" />
        </div>
      </div>
      <div class="field">
        <label class="label">Top K</label>
        <div class="control">
          <input class="input" id="topK" type="number" value="${config.topK}" />
        </div>
      </div>
      <div class="field">
        <label class="label">Document</label>
        <div class="control">
          <input class="input" type="file" id="file" accept=".txt" />
        </div>
      </div>
      <div class="field">
        <div class="control">
          <button class="button is-primary" type="submit">Upload</button>
        </div>
      </div>
    </form>
    <p class="has-text-weight-semibold" id="status"></p>
    <p><a href="/chat">Go to Chat</a></p>
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
  `);
}

function chatHtml() {
  return pageTemplate(`
    <h1 class="title">Chatbot</h1>
    <div id="messages" class="content" style="min-height: 200px;"></div>
    <div class="field has-addons">
      <div class="control is-expanded">
        <input class="input" id="msg" placeholder="Ask something..." />
      </div>
      <div class="control">
        <button class="button is-link" id="send">Send</button>
      </div>
    </div>
    <p><a href="/admin">Back to Admin</a></p>
    <script>
        document.getElementById('send').addEventListener('click', async () => {
          const msgEl = document.getElementById('msg');
          const msg = msgEl.value;
          if(!msg) return;
          msgEl.value = '';
          const chat = document.getElementById('messages');
          chat.innerHTML += '<p><strong>You:</strong> '+msg+'</p>';
          try {
            const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ message: msg }) });
            const data = await res.json();
            if (!res.ok || data.error) {
              throw new Error(data.error);
            }
            chat.innerHTML += '<p><strong>Bot:</strong> '+data.answer+'</p>';
          } catch (e) {
            chat.innerHTML += '<p><strong>Bot:</strong> Failed to generate answer</p>';
          }
        });
    </script>
  `);
}

function homeHtml() {
  return pageTemplate(`
    <h1 class="title">RAG Chatbot Demo</h1>
    <div class="buttons">
      <a class="button is-link" href="/admin">Admin</a>
      <a class="button is-primary" href="/chat">Chat</a>
    </div>
  `);
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

app.get('/', (req, res) => {
  res.send(homeHtml());
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Server running on port', PORT));
