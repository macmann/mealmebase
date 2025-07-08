require('dotenv').config();
const express = require('express');
const axios = require('axios');
const { CohereEmbeddings } = require('@langchain/cohere');
const { QdrantClient } = require('@qdrant/js-client-rest');

const app = express();
app.use(express.json());

const config = {
  instruction: '',
  temperature: 0.7,
  topP: 1,
  topK: 3,
};

// --------- Qdrant Client Setup ---------
const qdrant = new QdrantClient({
  url: process.env.QDRANT_HOST || 'http://localhost:6333',
  apiKey: process.env.QDRANT_API_KEY, // only needed for Qdrant Cloud
});
const collection = 'docs';

// ------- Qdrant Collection Helper -------
async function ensureCollection() {
  try {
    await qdrant.createCollection(collection, { vectors: { size: 1536, distance: 'Cosine' } });
  } catch (e) {
    // Ignore if already exists
  }
}

async function ingestDocument(text) {
  try {
    const embeddings = new CohereEmbeddings({
      apiKey: process.env.COHERE_API_KEY,
      model: "embed-v4.0",
    });
    await ensureCollection();
    const vector = await embeddings.embedQuery(text);
    await qdrant.upsert(collection, {
      points: [{ id: Date.now(), vector, payload: { text } }],
    });
    console.log('Ingested document:', text.slice(0, 40) + '...');
    return true;
  } catch (e) {
    console.error('Ingest error:', e);
    return false;
  }
}

async function searchDocs(query) {
  try {
    const embeddings = new CohereEmbeddings({
      apiKey: process.env.COHERE_API_KEY,
      model: "embed-v4.0",
    });
    const vector = await embeddings.embedQuery(query);
    const results = await qdrant.search(collection, {
      vector,
      limit: config.topK,
      with_payload: true,
    });
    console.log('Search returned', results.length, 'results');
    return results.map((r) => r.payload.text).join('\n');
  } catch (e) {
    console.error('Search error:', e);
    return '';
  }
}

async function askLLM(context, question) {
  try {
    const res = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-4.1-mini',
        messages: [
          { role: 'system', content: config.instruction || 'You are a helpful assistant.' },
          { role: 'user', content: context ? `${context}\n\n${question}` : question },
        ],
        temperature: config.temperature,
        top_p: config.topP,
      },
      { headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` } }
    );
    return res.data.choices[0].message.content.trim();
  } catch (e) {
    console.error('LLM error:', e.response ? e.response.data : e);
    return 'Sorry, could not generate an answer.';
  }
}

// ----- UI Layout -----
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
          .chat-box { min-height: 0; max-height: 100%; }
        </style>
      </head>
      <body class="bg-gray-100 min-h-screen">
        <section class="min-h-screen w-full flex flex-col px-2 py-4">
          <div class="flex-1 flex flex-col w-full">${content}</div>
        </section>
      </body>
    </html>
  `;
}

function adminHtml() {
  return pageTemplate(`
    <h1 class="text-3xl font-bold text-center mb-6">Admin & Test</h1>
    <div class="flex flex-col md:flex-row gap-6 flex-1 w-full">
      <div class="bg-white p-6 rounded shadow flex-1 flex flex-col min-w-[340px] md:min-w-[380px] w-full">
        <form id="upload-form" class="space-y-4 flex-1 flex flex-col w-full">
          <label class="block font-semibold mb-1">Instruction</label>
          <div class="mb-8">
            <div id="instruction-editor" class="h-40 w-full border rounded"></div>
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Temperature</label>
            <input class="w-full border rounded px-3 py-2" id="temperature" type="number" step="0.1" value="${config.temperature}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Top P</label>
            <input class="w-full border rounded px-3 py-2" id="topP" type="number" step="0.1" value="${config.topP}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Top K</label>
            <input class="w-full border rounded px-3 py-2" id="topK" type="number" value="${config.topK}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Document</label>
            <input class="w-full border rounded px-3 py-2" type="file" id="file" accept=".txt,.pdf" />
          </div>
          <div class="w-full">
            <button class="bg-blue-500 text-white px-4 py-2 rounded w-full" type="submit">Upload</button>
          </div>
          <p class="font-semibold" id="status"></p>
        </form>
      </div>
      <div class="bg-white p-6 rounded shadow flex flex-col flex-1 min-h-[500px] w-full">
        <h2 class="text-xl font-semibold mb-4">Test Chat</h2>
        <div id="messages" class="chat-box flex-1 overflow-y-auto space-y-2 mb-4"></div>
        <div class="flex gap-2">
          <input class="flex-1 border rounded-l px-3 py-3" id="msg" placeholder="Ask something..." />
          <button class="bg-blue-500 text-white px-5 py-3 rounded-r" id="send">Send</button>
        </div>
      </div>
    </div>
    <style>
      #instruction-editor, .ql-container, .ql-editor {
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box;
      }
      .ql-container {
        min-height: 8rem;
      }
      .mb-8 {
        margin-bottom: 2rem !important;
      }
    </style>
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
            txt += content.items.map(it => it.str).join(' ') + '\\n';
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
    <h1 class="text-3xl font-bold text-center mb-8">Chatbot</h1>
    <div class="bg-white rounded shadow p-4 flex flex-col h-[75vh] w-full">
      <div id="messages" class="chat-box flex-1 overflow-y-auto space-y-2 mb-4"></div>
      <div class="flex gap-2">
        <input class="flex-1 border rounded-l px-3 py-3" id="msg" placeholder="Ask something..." />
        <button class="bg-blue-500 text-white px-5 py-3 rounded-r" id="send">Send</button>
      </div>
      <p class="text-center mt-4"><a class="text-blue-500 underline" href="/admin">Back to Admin</a></p>
    </div>
    <style>
      .chat-box { min-height: 0; max-height: 100%; }
    </style>
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
    <h1 class="text-3xl font-bold mb-8 text-center">RAG Chatbot Demo</h1>
    <div class="flex justify-center space-x-4 w-full">
      <a class="bg-blue-500 text-white px-4 py-2 rounded" href="/admin">Admin</a>
      <a class="bg-green-500 text-white px-4 py-2 rounded" href="/chat">Chat</a>
    </div>
  `);
}

// ---- Routes ----
app.get('/admin', (req, res) => {
  res.send(adminHtml());
});

app.post('/admin', async (req, res) => {
  console.log('ADMIN POST', req.body);
  const { instruction, temperature, topP, topK, text } = req.body;
  if (instruction !== undefined) config.instruction = instruction;
  if (!isNaN(temperature)) config.temperature = temperature;
  if (!isNaN(topP)) config.topP = topP;
  if (!isNaN(topK)) config.topK = topK;
  if (text) {
    const ok = await ingestDocument(text);
    if (!ok) return res.status(500).json({ error: 'Ingest failed.' });
  }
  res.json({ status: 'ok' });
});

app.get('/chat', (req, res) => {
  res.send(chatHtml());
});

app.post('/chat', async (req, res) => {
  console.log('CHAT POST', req.body);
  const { message } = req.body;
  try {
    const context = await searchDocs(message);
    const answer = await askLLM(context, message);
    res.json({ answer });
  } catch (e) {
    console.error('Chat error:', e);
    res.status(500).json({ error: 'Failed to generate answer' });
  }
});

app.get('/', (req, res) => {
  res.send(homeHtml());
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Server running on port', PORT));

// --- Telegram Bot Integration ---
if (process.env.TELEGRAM_BOT_TOKEN) {
  const TelegramBot = require('node-telegram-bot-api');
  const bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });

  bot.on('message', async (msg) => {
    const chatId = msg.chat.id;
    const text = msg.text?.trim();
    if (!text) return;
    try {
      const context = await searchDocs(text);
      const answer = await askLLM(context, text);
      await bot.sendMessage(chatId, answer);
    } catch (e) {
      console.error('Telegram bot error:', e);
      await bot.sendMessage(chatId, 'Failed to generate answer');
    }
  });

  console.log('Telegram bot started');
} else {
  console.log('TELEGRAM_BOT_TOKEN not set, skipping Telegram bot startup');
}
// --- End of Telegram Bot Integration ---