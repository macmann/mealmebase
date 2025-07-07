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
        <script src="https://cdn.tailwindcss.com"></script>
        <title>RAG Chatbot</title>
        <style>
          .chat-box { min-height: 300px; max-height: 60vh; overflow-y: auto; }
        </style>
      </head>
      <body class="bg-gray-100">
        <section class="py-8">
          <div class="max-w-xl mx-auto">
            ${content}
          </div>
        </section>
      </body>
    </html>
  `;
}

function adminHtml() {
  return pageTemplate(`
    <h1 class="text-2xl font-bold text-center mb-6">Admin & Test</h1>
    <div class="flex flex-col md:flex-row gap-6">
      <div class="bg-white p-6 rounded shadow flex-1">
        <form id="upload-form" class="space-y-4">
          <div>
            <label class="block font-semibold mb-1">Instruction</label>
            <div id="instruction-editor" class="h-32"></div>
          </div>
          <div>
            <label class="block font-semibold mb-1">Temperature</label>
            <input class="w-full border rounded px-3 py-2" id="temperature" type="number" step="0.1" value="${config.temperature}" />
          </div>
          <div>
            <label class="block font-semibold mb-1">Top P</label>
            <input class="w-full border rounded px-3 py-2" id="topP" type="number" step="0.1" value="${config.topP}" />
          </div>
          <div>
            <label class="block font-semibold mb-1">Top K</label>
            <input class="w-full border rounded px-3 py-2" id="topK" type="number" value="${config.topK}" />
          </div>
          <div>
            <label class="block font-semibold mb-1">Document</label>
            <input class="w-full border rounded px-3 py-2" type="file" id="file" accept=".txt,.pdf" />
          </div>
          <div>
            <button class="bg-blue-500 text-white px-4 py-2 rounded" type="submit">Upload</button>
          </div>
          <p class="font-semibold" id="status"></p>
        </form>
      </div>
      <div class="bg-white p-6 rounded shadow w-full md:w-1/2">
        <h2 class="text-xl font-semibold mb-4">Test Chat</h2>
        <div id="messages" class="chat-box space-y-2 h-60"></div>
        <div class="flex mt-4">
          <input class="flex-1 border rounded-l px-3 py-2" id="msg" placeholder="Ask something..." />
          <button class="bg-blue-500 text-white px-4 py-2 rounded-r" id="send">Send</button>
        </div>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.min.js"></script>
    <link href="https://cdn.quilljs.com/1.3.6/quill.snow.css" rel="stylesheet">
    <script src="https://cdn.quilljs.com/1.3.6/quill.min.js"></script>
    <script>
      pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.14.305/pdf.worker.min.js';
      const quill = new Quill('#instruction-editor', { theme: 'snow' });
      quill.root.innerHTML = ${JSON.stringify(config.instruction)};

      async function fileToText(file) {
        if (!file) return '';
        if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
          const typedArray = new Uint8Array(await file.arrayBuffer());
          const pdf = await pdfjsLib.getDocument(typedArray).promise;
          let txt = '';
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            txt += content.items.map(it => it.str).join(' ') + '\n';
          }
          return txt;
        }
        return await file.text();
      }

      document.getElementById('upload-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const f = document.getElementById('file').files[0];
        const text = await fileToText(f);
        const body = {
          instruction: quill.root.innerHTML,
          temperature: parseFloat(document.getElementById('temperature').value),
          topP: parseFloat(document.getElementById('topP').value),
          topK: parseInt(document.getElementById('topK').value, 10),
          text,
        };
        const res = await fetch('/admin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        document.getElementById('status').innerText = res.ok ? 'Uploaded!' : 'Upload failed';
      });

      function appendMessage(role, text) {
        const cls = role === 'user' ? 'bg-blue-100' : 'bg-green-100';
        const chat = document.getElementById('messages');
        chat.innerHTML += '<div class="' + cls + ' rounded p-2"><strong>' + (role === 'user' ? 'You' : 'Bot') + ':</strong> ' + text + '</div>';
        chat.scrollTop = chat.scrollHeight;
      }

      async function sendMessage() {
        const msgEl = document.getElementById('msg');
        const msg = msgEl.value.trim();
        if (!msg) return;
        msgEl.value = '';
        appendMessage('user', msg);
        try {
          const res = await fetch('/chat', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ message: msg })
          });
          const data = await res.json();
          if (!res.ok || data.error) throw new Error(data.error);
          appendMessage('bot', data.answer);
        } catch (e) {
          appendMessage('bot', 'Failed to generate answer');
        }
      }

      document.getElementById('send').addEventListener('click', sendMessage);
      document.getElementById('msg').addEventListener('keydown', (e) => { if(e.key === 'Enter'){ e.preventDefault(); sendMessage(); }});
    </script>
  `);
}

function chatHtml() {
  return pageTemplate(`
    <h1 class="text-2xl font-bold text-center mb-6">Chatbot</h1>
    <div class="bg-white p-6 rounded shadow">
      <div id="messages" class="chat-box space-y-2"></div>
      <div class="flex mt-4">
        <input class="flex-1 border rounded-l px-3 py-2" id="msg" placeholder="Ask something..." />
        <button class="bg-blue-500 text-white px-4 py-2 rounded-r" id="send">Send</button>
      </div>
      <p class="text-center mt-4"><a class="text-blue-500 underline" href="/admin">Back to Admin</a></p>
    </div>
    <script>
        function appendMessage(role, text) {
          const cls = role === 'user' ? 'bg-blue-100' : 'bg-green-100';
          const chat = document.getElementById('messages');
          chat.innerHTML += '<div class="' + cls + ' rounded p-2"><strong>' + (role === 'user' ? 'You' : 'Bot') + ':</strong> ' + text + '</div>';
          chat.scrollTop = chat.scrollHeight;
        }
        async function sendMessage() {
          const msgEl = document.getElementById('msg');
          const msg = msgEl.value.trim();
          if (!msg) return;
          msgEl.value = '';
          appendMessage('user', msg);
          try {
            const res = await fetch('/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ message: msg }) });
            const data = await res.json();
            if (!res.ok || data.error) throw new Error(data.error);
            appendMessage('bot', data.answer);
          } catch (e) {
            appendMessage('bot', 'Failed to generate answer');
          }
        }
        document.getElementById('send').addEventListener('click', sendMessage);
        document.getElementById('msg').addEventListener('keydown', (e) => { if(e.key === 'Enter'){ e.preventDefault(); sendMessage(); }});
    </script>
  `);
}

function homeHtml() {
  return pageTemplate(`
    <h1 class="text-2xl font-bold mb-6 text-center">RAG Chatbot Demo</h1>
    <div class="flex justify-center space-x-4">
      <a class="bg-blue-500 text-white px-4 py-2 rounded" href="/admin">Admin</a>
      <a class="bg-green-500 text-white px-4 py-2 rounded" href="/chat">Chat</a>
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
