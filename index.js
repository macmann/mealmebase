require('dotenv').config();
const express = require('express');
const axios = require('axios');
const { CohereEmbeddings } = require('@langchain/cohere');
const { QdrantClient } = require('@qdrant/js-client-rest');
const fs = require('fs');
const crypto = require('crypto');
const cookie = require('cookie');

const LOGO_URL = process.env.LOGO_URL || 'https://ibb.co/HDy3fYZ8';

const app = express();
// Increase body size limit to handle large document uploads
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: false }));

const sessions = {};

app.use((req, res, next) => {
  const sid = cookie.parse(req.headers.cookie || '').sid;
  if (sid && sessions[sid] && users[sessions[sid]]) {
    req.user = { email: sessions[sid], isAdmin: users[sessions[sid]].isAdmin };
    req.sessionId = sid;
  }
  next();
});

function hashPassword(pw) {
  return crypto.createHash('sha256').update(pw).digest('hex');
}

const AGENTS_FILE = 'agents.json';
let agents = {};

const USERS_FILE = 'users.json';
let users = {};
try {
  users = JSON.parse(fs.readFileSync(USERS_FILE, 'utf8'));
} catch {
  users = {};
}
if (!users['admin@mealme.com']) {
  users['admin@mealme.com'] = { password: hashPassword('admin'), isAdmin: true };
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));
}
function saveUsers() {
  fs.promises
    .writeFile(USERS_FILE, JSON.stringify(users, null, 2))
    .catch((e) => console.error('Users save error:', e));
}

try {
  agents = JSON.parse(fs.readFileSync(AGENTS_FILE, 'utf8'));
} catch {
  agents = {};
}

function saveAgents() {
  fs.promises
    .writeFile(AGENTS_FILE, JSON.stringify(agents, null, 2))
    .catch((e) => console.error('Agents save error:', e));
}


// Persisted chat history for Telegram bot
const HISTORY_FILE = 'chathistory.json';
// Structure: { [agentId]: { [chatId]: [{role,text}] } }
let telegramHistory = {};
const dashboardHistory = {};

try {
  telegramHistory = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
} catch {
  telegramHistory = {};
}

function saveHistory() {
  fs.promises
    .writeFile(HISTORY_FILE, JSON.stringify(telegramHistory, null, 2))
    .catch((e) => console.error('History save error:', e));
}

function addTelegramMessage(agentId, chatId, role, text) {
  if (!telegramHistory[agentId]) telegramHistory[agentId] = {};
  if (!telegramHistory[agentId][chatId]) telegramHistory[agentId][chatId] = [];
  telegramHistory[agentId][chatId].push({ role, text });
  if (telegramHistory[agentId][chatId].length > 5) {
    telegramHistory[agentId][chatId] = telegramHistory[agentId][chatId].slice(-5);
  }
  saveHistory();
}

function addDashboardMessage(sessionId, role, text) {
  if (!dashboardHistory[sessionId]) dashboardHistory[sessionId] = [];
  dashboardHistory[sessionId].push({ role, text });
  if (dashboardHistory[sessionId].length > 5) {
    dashboardHistory[sessionId] = dashboardHistory[sessionId].slice(-5);
  }
}

function requireLogin(req, res, next) {
  if (!req.user) {
    if (req.method === 'GET') return res.redirect('/login');
    return res.status(401).json({ error: 'Login required' });
  }
  next();
}

app.use((req, res, next) => {
  if (!req.user && req.path !== '/login' && !req.path.startsWith('/viber/webhook')) {
    if (req.method === 'GET') return res.redirect('/login');
    return res.status(401).json({ error: 'Login required' });
  }
  next();
});

// --------- Qdrant Client Setup ---------
const qdrant = new QdrantClient({
  url: process.env.QDRANT_HOST || 'http://localhost:6333',
  apiKey: process.env.QDRANT_API_KEY, // only needed for Qdrant Cloud
});


// ------- Qdrant Collection Helper -------
async function ensureCollection(name) {
  try {
    await qdrant.createCollection(name, { vectors: { size: 1536, distance: 'Cosine' } });
  } catch (e) {
    // Ignore if already exists
  }
}

async function ingestDocument(collection, text, name) {
  try {
    const embeddings = new CohereEmbeddings({
      apiKey: process.env.COHERE_API_KEY,
      model: "embed-v4.0",
    });
    await ensureCollection(collection);
    const vector = await embeddings.embedQuery(text);
    await qdrant.upsert(collection, {
      points: [{ id: Date.now() + Math.floor(Math.random() * 1e6), vector, payload: { text, name } }],
    });
    console.log('Ingested document:', text.slice(0, 40) + '...');
    return true;
  } catch (e) {
    console.error('Ingest error:', e);
    return false;
  }
}

async function searchDocs(collection, query, topK) {
  try {
    const embeddings = new CohereEmbeddings({
      apiKey: process.env.COHERE_API_KEY,
      model: "embed-v4.0",
    });
    const vector = await embeddings.embedQuery(query);
    const results = await qdrant.search(collection, {
      vector,
      limit: topK,
      with_payload: true,
    });
    console.log('Search returned', results.length, 'results');
    return results.map((r) => r.payload.text).join('\\n');
  } catch (e) {
    console.error('Search error:', e);
    return '';
  }
}

async function listDocs(collection) {
  await ensureCollection(collection);
  const docs = [];
  let offset = undefined;
  do {
    const res = await qdrant.scroll(collection, { limit: 50, offset, with_payload: true, with_vector: false });
    docs.push(...res.points.map(p => ({ id: p.id, name: p.payload?.name || 'Document' })));
    offset = res.next_page_offset;
  } while (offset != null);
  return docs;
}

async function removeDoc(collection, id) {
  try {
    const pointId = typeof id === 'string' ? Number(id) : id;
    await qdrant.delete(collection, { points: [pointId] });
    return true;
  } catch (e) {
    console.error('Delete error:', e);
    return false;
  }
}

async function askLLM(agent, context, question, history = []) {
  try {
    const res = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-4.1-mini',
        messages: [
          { role: 'system', content: agent.instruction || 'You are a helpful assistant.' },
          ...history.slice(-5).map(m => ({
            role: m.role === 'bot' ? 'assistant' : m.role,
            content: m.text,
          })),
          { role: 'user', content: context ? `${context}\n\n${question}` : question },
        ],
        temperature: agent.temperature,
        top_p: agent.topP,
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
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <title>dot.Me Platform</title>
        <style>
          .chat-box { min-height: 0; max-height: 100%; }
          .nav-btn {
            background-color: #3b82f6;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 0.25rem;
          }
        </style>
      </head>
      <body class="bg-gray-100 min-h-screen">
        <header class="bg-white shadow p-4 mb-4 flex items-center justify-between">
          <div class="flex items-center">
            <img src="${LOGO_URL}" alt="Logo" class="h-12 mr-3" />
            <span class="text-xl font-bold">dot.Me Platform</span>
          </div>
          <nav class="flex gap-4 ml-auto">
            <button class="nav-btn" onclick="location.href='/'">Home</button>
            <button class="nav-btn" onclick="location.href='/admin'">Admin</button>
            <button class="nav-btn" onclick="location.href='/chat'">Chat</button>
          </nav>
        </header>
        <section class="min-h-screen w-full flex flex-col px-2 py-4">
          <div class="flex-1 flex flex-col w-full">${content}</div>
        </section>
      </body>
    </html>
  `;
}


function adminHtml(agent) {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const threads = Object.entries(dashboardHistory).filter(([k]) => k.endsWith('-' + agent.id));
  const historySections = threads.map(([tid, msgs]) => {
    const msgHtml = msgs.map(m => `<div class="${m.role === 'user' ? 'bg-blue-100' : 'bg-green-100'} rounded p-2 mb-1"><strong>${m.role === 'user' ? 'User' : 'Bot'}:</strong> <span class="md">${esc(m.text)}</span></div>`).join('');
    return `<div class="border rounded p-4 mb-4"><h2 class="font-semibold mb-2">Thread ${tid}</h2>${msgHtml}</div>`;
  }).join('');
  return pageTemplate(`
    <h1 class="text-3xl font-bold text-center mb-2">Admin - ${agent.name}</h1>
    <div class="flex gap-2 mb-4 justify-center">
      <button id="tab-settings" class="tab bg-blue-500 text-white px-3 py-1 rounded">Settings</button>
      <button id="tab-history" class="tab bg-gray-200 px-3 py-1 rounded">Chat History</button>
    </div>
    <div id="settings-pane" class="tab-content flex flex-col md:flex-row gap-6 flex-1 w-full">
      <div class="bg-white p-6 rounded shadow flex-1 flex flex-col min-w-[340px] md:min-w-[380px] w-full">
        <form id="upload-form" class="space-y-4 flex-1 flex flex-col w-full">
          <div class="w-full">
            <label class="block font-semibold mb-1">Name</label>
            <input class="w-full border rounded px-3 py-2" id="name" value="${agent.name}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1" for="instruction">Instruction</label>
            <textarea id="instruction" class="w-full h-40 border rounded p-2">${esc(agent.instruction)}</textarea>
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Temperature</label>
            <input class="w-full border rounded px-3 py-2" id="temperature" type="number" step="0.1" value="${agent.temperature}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Top P</label>
            <input class="w-full border rounded px-3 py-2" id="topP" type="number" step="0.1" value="${agent.topP}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Top K</label>
            <input class="w-full border rounded px-3 py-2" id="topK" type="number" value="${agent.topK}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Telegram Bot Token</label>
            <input class="w-full border rounded px-3 py-2" id="telegramToken" value="${agent.telegramToken || ''}" />
          </div>
          <div class="w-full">
            <label class="block font-semibold mb-1">Documents</label>
            <input class="w-full border rounded px-3 py-2" type="file" id="file" accept=".txt,.pdf,.json,.csv" multiple />
          </div>
          <div class="w-full">
            <button class="bg-blue-500 text-white px-4 py-2 rounded w-full" type="submit">Upload</button>
          </div>
          <p class="font-semibold" id="status"></p>
        </form>
        <h2 class="text-lg font-semibold mt-6">Existing Documents</h2>
        <div id="docs" class="mt-2 space-y-2"></div>
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
    <div id="history-pane" class="tab-content hidden">
      ${historySections || '<p>No history yet</p>'}
    </div>
    <style>
      #instruction { width: 100%; box-sizing: border-box; }
      .mb-8 { margin-bottom: 2rem !important; }
      .hidden { display: none; }
    </style>
    <script>
      const instructionEl = document.getElementById('instruction');
      document.getElementById('tab-settings').addEventListener('click', () => {
        document.getElementById('settings-pane').classList.remove('hidden');
        document.getElementById('history-pane').classList.add('hidden');
        document.getElementById('tab-settings').classList.add('bg-blue-500','text-white');
        document.getElementById('tab-history').classList.remove('bg-blue-500','text-white');
      });
      document.getElementById('tab-history').addEventListener('click', () => {
        document.getElementById('history-pane').classList.remove('hidden');
        document.getElementById('settings-pane').classList.add('hidden');
        document.getElementById('tab-history').classList.add('bg-blue-500','text-white');
        document.getElementById('tab-settings').classList.remove('bg-blue-500','text-white');
        document.querySelectorAll('.md').forEach(el => { el.innerHTML = marked.parse(el.textContent); });
      });
      async function fileToText(file) {
        if (!file) return '';
        if (file.type === 'application/json' || file.name.toLowerCase().endsWith('.json')) {
          const raw = await file.text();
          try {
            const data = JSON.parse(raw);
            return JSON.stringify(data, null, 2);
          } catch (e) {
            console.error('Invalid JSON file', e);
            return raw;
          }
        }
        if (file.type === 'text/csv' || file.name.toLowerCase().endsWith('.csv')) {
          return await file.text();
        }
        return await file.text();
      }
      document.getElementById('upload-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const files = [...document.getElementById('file').files];
        const docs = [];
        for (const f of files) {
          const text = await fileToText(f);
          docs.push({ name: f.name, text });
        }
        const body = {
          name: document.getElementById('name').value.trim(),
          instruction: instructionEl.value,
          temperature: parseFloat(document.getElementById('temperature').value),
          topP: parseFloat(document.getElementById('topP').value),
          topK: parseInt(document.getElementById('topK').value, 10),
          telegramToken: document.getElementById('telegramToken').value.trim(),
          files: docs,
        };
        const res = await fetch('/admin/${agent.id}', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        document.getElementById('status').innerText = res.ok ? 'Uploaded!' : 'Upload failed';
        if (res.ok) loadDocs();
      });
      function appendMessage(role, text) {
        const cls = role === 'user' ? 'bg-blue-100' : 'bg-green-100';
        const chat = document.getElementById('messages');
        const html = marked.parse(text);
        chat.innerHTML += '<div class="' + cls + ' rounded p-2"><strong>' + (role === 'user' ? 'You' : 'Bot') + ':</strong> ' + html + '</div>';
        chat.scrollTop = chat.scrollHeight;
      }
      async function sendMessage() {
        const msgEl = document.getElementById('msg');
        const msg = msgEl.value.trim();
        if (!msg) return;
        msgEl.value = '';
        appendMessage('user', msg);
        try {
          const res = await fetch('/chat/${agent.id}', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ message: msg }) });
          const data = await res.json();
          if (!res.ok || data.error) throw new Error(data.error);
          appendMessage('bot', data.answer);
        } catch (e) {
          appendMessage('bot', 'Failed to generate answer');
        }
      }
      document.getElementById('send').addEventListener('click', sendMessage);
      document.getElementById('msg').addEventListener('keydown', (e) => { if(e.key === 'Enter'){ e.preventDefault(); sendMessage(); }});
      async function loadDocs() {
        const res = await fetch('/docs/${agent.id}');
        if (!res.ok) return;
        const docs = await res.json();
        const container = document.getElementById('docs');
        container.innerHTML = '';
        docs.forEach(d => {
          const div = document.createElement('div');
          div.className = 'flex justify-between items-center border rounded p-2';
          div.innerHTML = '<span>' + d.name + '</span>' + '<button data-id="' + d.id + '" class="delete bg-red-500 text-white px-2 rounded">Delete</button>';
          container.appendChild(div);
        });
      }
      document.getElementById('docs').addEventListener('click', async (e) => {
        if (e.target.classList.contains('delete')) {
          const id = e.target.getAttribute('data-id');
          const res = await fetch('/docs/${agent.id}/' + id, { method: 'DELETE' });
          if (res.ok) loadDocs();
        }
      });
      loadDocs();
    </script>
  `);
}
function chatHtml(agent) {
  return pageTemplate(`
    <h1 class="text-3xl font-bold text-center mb-8">Chatbot - ${agent.name}</h1>
    <div class="bg-white rounded shadow p-4 flex flex-col h-[75vh] w-full">
      <div id="messages" class="chat-box flex-1 overflow-y-auto space-y-2 mb-4"></div>
      <div class="flex gap-2">
        <input class="flex-1 border rounded-l px-3 py-3" id="msg" placeholder="Ask something..." />
        <button class="bg-blue-500 text-white px-5 py-3 rounded-r" id="send">Send</button>
      </div>
      <p class="text-center mt-4"><a class="text-blue-500 underline" href="/admin/${agent.id}">Back to Admin</a></p>
    </div>
    <style>
      .chat-box { min-height: 0; max-height: 100%; }
    </style>
    <script>
      function appendMessage(role, text) {
        const cls = role === 'user' ? 'bg-blue-100' : 'bg-green-100';
        const chat = document.getElementById('messages');
        const html = marked.parse(text);
        chat.innerHTML += '<div class="' + cls + ' rounded p-2"><strong>' + (role === 'user' ? 'You' : 'Bot') + ':</strong> ' + html + '</div>';
        chat.scrollTop = chat.scrollHeight;
      }
      async function sendMessage() {
        const msgEl = document.getElementById('msg');
        const msg = msgEl.value.trim();
        if (!msg) return;
        msgEl.value = '';
        appendMessage('user', msg);
        try {
          const res = await fetch('/chat/${agent.id}', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ message: msg }) });
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

function historyHtml() {
  const esc = (s) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const sections = Object.entries(telegramHistory).map(([agentId, chats]) => {
    const agentName = agents[agentId]?.name || agentId;
    const chatsHtml = Object.entries(chats).map(([cid, msgs]) => {
      const msgHtml = msgs
        .map(m => `<div class="${m.role === 'user' ? 'bg-blue-100' : 'bg-green-100'} rounded p-2 mb-1"><strong>${m.role === 'user' ? 'User' : 'Bot'}:</strong> <span class="md">${esc(m.text)}</span></div>`)
        .join('');
      return `<div class="border rounded p-2 mb-2"><h3 class="font-semibold mb-1">Chat ${cid}</h3>${msgHtml}</div>`;
    }).join('');
    return `<div class="border-2 rounded p-4 mb-4"><h2 class="font-semibold mb-2">Agent ${agentName}</h2>${chatsHtml}</div>`;
  }).join('');
  return pageTemplate(`
    <h1 class="text-3xl font-bold text-center mb-8">Telegram Chat History</h1>
    <div class="overflow-y-auto flex-1">${sections || '<p>No history yet</p>'}</div>
    <p class="text-center mt-4"><a class="text-blue-500 underline" href="/">Home</a></p>
    <script>
      document.querySelectorAll('.md').forEach(el => {
        el.innerHTML = marked.parse(el.textContent);
      });
    </script>
  `);
}

function homeHtml() {
  return pageTemplate(`
    <h1 class="text-3xl font-bold mb-8 text-center">dot.Me Platform Demo</h1>
    <div class="flex justify-center gap-8 w-full">
      <a href="/admin" class="block bg-white shadow rounded p-4 w-40 text-center hover:bg-gray-100">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l.7 2.152a1 1 0 00.95.69h2.252c.969 0 1.371 1.24.588 1.81l-1.823 1.322a1 1 0 00-.364 1.118l.7 2.152c.3.921-.755 1.688-1.54 1.118l-1.823-1.322a1 1 0 00-1.176 0l-1.823 1.322c-.784.57-1.838-.197-1.539-1.118l.7-2.152a1 1 0 00-.364-1.118L4.21 7.579c-.783-.57-.38-1.81.588-1.81h2.252a1 1 0 00.95-.69l.7-2.152z" />
        </svg>
        <span class="block font-semibold">Admin</span>
      </a>
      <a href="/chat" class="block bg-white shadow rounded p-4 w-40 text-center hover:bg-gray-100">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8a9 9 0 100-18 9 9 0 000 18z" />
        </svg>
        <span class="block font-semibold">User</span>
      </a>
    </div>
  `);
}

function loginHtml(error = '') {
  return pageTemplate(`
    <h1 class="text-3xl font-bold mb-4 text-center">Login</h1>
    <form method="POST" class="max-w-sm mx-auto space-y-4">
      ${error ? `<p class="text-red-500">${error}</p>` : ''}
      <input class="w-full border px-3 py-2" name="email" placeholder="Email" />
      <input class="w-full border px-3 py-2" name="password" type="password" placeholder="Password" />
      <button class="bg-blue-500 text-white px-4 py-2 rounded w-full" type="submit">Login</button>
    </form>
  `);
}

function usersHtml() {
  const userList = Object.entries(users).map(([email, u]) =>
    `<div class="flex justify-between items-center border rounded p-2">
      <span>${email}</span>
      ${email !== 'admin@mealme.com' ? `<button data-email="${email}" class="del bg-red-500 text-white px-2 rounded">Delete</button>` : ''}
    </div>`
  ).join('');
  return pageTemplate(`
    <h1 class="text-3xl font-bold mb-4 text-center">Manage Users</h1>
    <div class="space-y-2 mb-4">${userList || '<p>No users</p>'}</div>
    <form id="createForm" class="flex flex-col gap-2 max-w-sm mx-auto">
      <input class="border px-2 py-1" name="email" placeholder="Email" />
      <input class="border px-2 py-1" name="password" type="password" placeholder="Password" />
      <label class="inline-flex items-center"><input type="checkbox" name="isAdmin" class="mr-2" />Admin</label>
      <button class="bg-blue-500 text-white px-3 py-1 rounded" type="submit">Create</button>
    </form>
    <p class="text-center mt-4"><a class="text-blue-500 underline" href="/">Home</a></p>
    <script>
      document.getElementById('createForm').addEventListener('submit', async e => {
        e.preventDefault();
        const fd = new FormData(e.target);
        const res = await fetch('/users', { method:'POST', body: new URLSearchParams(fd) });
        if(res.ok) location.reload();
      });
      document.querySelectorAll('.del').forEach(btn => {
        btn.addEventListener('click', async () => {
          const email = btn.getAttribute('data-email');
          if(!confirm('Delete ' + email + '?')) return;
          const res = await fetch('/users/' + encodeURIComponent(email), { method:'DELETE' });
          if(res.ok) location.reload();
        });
      });
    </script>
  `);
}

function changePasswordHtml(msg = '') {
  return pageTemplate(`
    <h1 class="text-3xl font-bold mb-4 text-center">Change Password</h1>
    ${msg ? `<p class="text-green-600 text-center">${msg}</p>` : ''}
    <form method="POST" class="max-w-sm mx-auto space-y-4">
      <input class="w-full border px-3 py-2" name="password" type="password" placeholder="New password" />
      <button class="bg-blue-500 text-white px-4 py-2 rounded w-full" type="submit">Change</button>
    </form>
    <p class="text-center mt-4"><a class="text-blue-500 underline" href="/">Home</a></p>
  `);
}

function adminListHtml() {
  const list = Object.values(agents).map(a => `
    <div class="flex justify-between items-center border rounded p-2">
      <span>${a.name}</span>
      <span>
        <a class="text-blue-500 underline" href="/admin/${a.id}">Edit</a>
      </span>
    </div>
  `).join('');
  return pageTemplate(`
    <h1 class="text-3xl font-bold mb-4 text-center">Manage Agents</h1>
    <div class="space-y-2 mb-4">${list || '<p>No agents</p>'}</div>
    <form id="create-form" class="flex flex-col md:flex-row gap-2 justify-center">
      <input class="border px-2 py-1 flex-1" id="name" placeholder="New agent name" />
      <input class="border px-2 py-1 flex-1" id="telegramToken" placeholder="Telegram bot token (optional)" />
      <button class="bg-blue-500 text-white px-3 py-1 rounded" type="submit">Create</button>
    </form>
    <p class="text-center mt-4"><a class="text-blue-500 underline" href="/">Home</a></p>
    <script>
      document.getElementById('create-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('name').value.trim();
        const token = document.getElementById('telegramToken').value.trim();
        if(!name) return;
        const res = await fetch('/agents', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name, telegramToken: token})});
        if(res.ok) location.reload();
      });
    </script>
  `);
}

function chatPanelHtml() {
  const agentArr = Object.values(agents);
  const first = agentArr[0] || { id: '', name: '' };
  return pageTemplate(`
    <div class="flex flex-col md:flex-row gap-4 flex-1 w-full">
      <div class="bg-white p-4 rounded shadow md:w-1/4 overflow-y-auto" id="sidebar">
        <h2 class="text-xl font-semibold mb-2">Agents</h2>
        <div id="agent-list" class="space-y-2"></div>
        <button id="view-history" class="mt-4 bg-purple-500 text-white px-3 py-1 rounded w-full">Chat History</button>
      </div>
      <div class="bg-white p-4 rounded shadow flex flex-col flex-1 min-h-[500px]">
        <h2 id="agent-name" class="text-xl font-semibold mb-2"></h2>
        <div id="messages" class="chat-box flex-1 overflow-y-auto space-y-2 mb-4"></div>
        <div class="flex gap-2">
          <input class="flex-1 border rounded-l px-3 py-3" id="msg" placeholder="Ask something..." />
          <button class="bg-blue-500 text-white px-5 py-3 rounded-r" id="send">Send</button>
        </div>
      </div>
    </div>
    <style>
      .chat-box { min-height: 0; max-height: 100%; }
    </style>
    <script>
      const agents = ${JSON.stringify(agentArr)};
      let current = '${first.id}';

      function renderAgents() {
        const listEl = document.getElementById('agent-list');
        listEl.innerHTML = '';
        agents.forEach(a => {
          const div = document.createElement('div');
          div.className = 'agent-item flex justify-between items-center cursor-pointer p-2 rounded ' + (a.id === current ? 'bg-blue-100' : 'bg-gray-100');
          const nameSpan = document.createElement('span');
          nameSpan.textContent = a.name;
          nameSpan.addEventListener('click', () => {
            current = a.id;
            clearMessages();
            renderAgents();
            document.getElementById('agent-name').textContent = a.name;
          });
          const settings = document.createElement('button');
          settings.innerHTML = '&#9881;';
          settings.className = 'settings text-gray-500';
          settings.addEventListener('click', (e) => {
            e.stopPropagation();
            window.location.href = '/admin/' + a.id;
          });
          div.appendChild(nameSpan);
          div.appendChild(settings);
          listEl.appendChild(div);
        });
      }

      function clearMessages() { document.getElementById('messages').innerHTML = ''; }

      function appendMessage(role, text) {
        const cls = role === 'user' ? 'bg-blue-100' : 'bg-green-100';
        const chat = document.getElementById('messages');
        const html = marked.parse(text);
        chat.innerHTML += '<div class="' + cls + ' rounded p-2"><strong>' + (role === 'user' ? 'You' : 'Bot') + ':</strong> ' + html + '</div>';
        chat.scrollTop = chat.scrollHeight;
      }

      async function sendMessage() {
        const msgEl = document.getElementById('msg');
        const msg = msgEl.value.trim();
        if (!msg || !current) return;
        msgEl.value = '';
        appendMessage('user', msg);
        try {
          const res = await fetch('/chat/' + current, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ message: msg }) });
          const data = await res.json();
          if (!res.ok || data.error) throw new Error(data.error);
          appendMessage('bot', data.answer);
        } catch (e) {
          appendMessage('bot', 'Failed to generate answer');
        }
      }

      document.getElementById('send').addEventListener('click', sendMessage);
      document.getElementById('msg').addEventListener('keydown', (e) => { if(e.key === 'Enter'){ e.preventDefault(); sendMessage(); }});
      document.getElementById('view-history').addEventListener('click', () => { if(current) window.location.href = '/user-history/' + current; });

      if (first) { document.getElementById('agent-name').textContent = first.name; }
      renderAgents();
    </script>
  `);
}

function userHistoryHtml(id) {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const threads = Object.entries(dashboardHistory).filter(([k]) => k.endsWith('-' + id));
  const sections = threads.map(([tid, msgs]) => {
    const msgHtml = msgs.map(m => `<div class="${m.role === 'user' ? 'bg-blue-100' : 'bg-green-100'} rounded p-2 mb-1"><strong>${m.role === 'user' ? 'User' : 'Bot'}:</strong> <span class="md">${esc(m.text)}</span></div>`).join('');
    return `<div class="border rounded p-4 mb-4"><h2 class="font-semibold mb-2">Thread ${tid}</h2>${msgHtml}</div>`;
  }).join('');
  const name = agents[id]?.name || 'Agent';
  return pageTemplate(`
    <h1 class="text-3xl font-bold text-center mb-8">Chat History - ${name}</h1>
    <div class="overflow-y-auto flex-1">${sections || '<p>No history yet</p>'}</div>
    <p class="text-center mt-4"><a class="text-blue-500 underline" href="/chat">Back</a></p>
    <script>
      document.querySelectorAll('.md').forEach(el => { el.innerHTML = marked.parse(el.textContent); });
    </script>
  `);
}

// ---- Routes ----
app.get('/login', (req, res) => {
  if (req.user) return res.redirect('/');
  res.send(loginHtml(req.query.error));
});

app.post('/login', (req, res) => {
  const { email, password } = req.body;
  const user = users[email];
  if (user && user.password === hashPassword(password)) {
    const sid = crypto.randomBytes(16).toString('hex');
    sessions[sid] = email;
    res.setHeader('Set-Cookie', cookie.serialize('sid', sid, { httpOnly: true, path: '/' }));
    return res.redirect('/');
  }
  res.redirect('/login?error=Invalid%20credentials');
});

app.get('/logout', (req, res) => {
  if (req.sessionId) delete sessions[req.sessionId];
  res.setHeader('Set-Cookie', cookie.serialize('sid', '', { path: '/', expires: new Date(0) }));
  res.redirect('/login');
});

app.get('/change-password', requireLogin, (req, res) => {
  res.send(changePasswordHtml());
});

app.post('/change-password', requireLogin, (req, res) => {
  const { password } = req.body;
  if (!password) return res.redirect('/change-password');
  const email = req.user.email;
  users[email].password = hashPassword(password);
  saveUsers();
  res.send(changePasswordHtml('Password updated'));
});

app.get('/users', requireLogin, (req, res) => {
  if (!req.user.isAdmin) return res.status(403).send('Forbidden');
  res.send(usersHtml());
});

app.post('/users', requireLogin, (req, res) => {
  if (!req.user.isAdmin) return res.status(403).send('Forbidden');
  const { email, password, isAdmin } = req.body;
  if (!email || !password || users[email]) return res.status(400).send('Invalid');
  users[email] = { password: hashPassword(password), isAdmin: Boolean(isAdmin) };
  saveUsers();
  res.sendStatus(200);
});

app.delete('/users/:email', requireLogin, (req, res) => {
  if (!req.user.isAdmin) return res.status(403).send('Forbidden');
  const email = decodeURIComponent(req.params.email);
  if (email === 'admin@mealme.com') return res.status(400).send('Cannot delete');
  delete users[email];
  saveUsers();
  res.sendStatus(200);
});

app.get('/agents', (req, res) => {
  res.send(adminListHtml());
});

app.post('/agents', async (req, res) => {
  const { name, telegramToken } = req.body;
  const id = 'a' + Date.now();
  const collection = `agent_${id}`;
  agents[id] = { id, name: name || 'Agent', instruction: '', temperature: 0.7, topP: 1, topK: 3, collection, telegramToken: telegramToken || '' };
  try {
    await ensureCollection(collection);
    saveAgents();
    if (agents[id].telegramToken) startTelegramBot(agents[id]);
    res.json({ id });
  } catch (e) {
    console.error('Create agent error:', e);
    res.status(500).json({ error: 'Failed to create agent' });
  }
});

app.get('/admin', (req, res) => {
  res.send(adminListHtml());
});

app.get('/chat', (req, res) => {
  res.send(chatPanelHtml());
});

app.get('/admin/:id', (req, res) => {
  const agent = agents[req.params.id];
  if (!agent) return res.status(404).send('Agent not found');
  res.send(adminHtml(agent));
});

app.post('/admin/:id', async (req, res) => {
  const agent = agents[req.params.id];
  if (!agent) return res.status(404).json({ error: 'Agent not found' });
  console.log('ADMIN POST', req.body);
  const { name, instruction, temperature, topP, topK, telegramToken, files = [], text } = req.body;
  if (typeof name === 'string' && name.trim()) agent.name = name.trim();
  if (instruction !== undefined) agent.instruction = instruction;
  if (!isNaN(temperature)) agent.temperature = temperature;
  if (!isNaN(topP)) agent.topP = topP;
  if (!isNaN(topK)) agent.topK = topK;
  if (telegramToken !== undefined) agent.telegramToken = telegramToken;
  if (files && Array.isArray(files)) {
    for (const f of files) {
      if (!f || !f.text) continue;
      let txt = f.text;
      if (f.name && f.name.toLowerCase().endsWith('.json')) {
        try {
          const data = JSON.parse(txt);
          txt = JSON.stringify(data);
        } catch (e) {
          console.error('Invalid JSON file', f.name, e);
          return res.status(400).json({ error: 'Invalid JSON file' });
        }
      }
      const ok = await ingestDocument(agent.collection, txt, f.name);
      if (!ok) return res.status(500).json({ error: 'Ingest failed.' });
    }
  } else if (text) {
    const ok = await ingestDocument(agent.collection, text, 'Document');
    if (!ok) return res.status(500).json({ error: 'Ingest failed.' });
  }
  saveAgents();
  if (agent.telegramToken) startTelegramBot(agent); else stopTelegramBot(agent.id);
  res.json({ status: 'ok' });
});

app.get('/docs/:agentId', async (req, res) => {
  const agent = agents[req.params.agentId];
  if (!agent) return res.status(404).json({ error: 'Agent not found' });
  try {
    const docs = await listDocs(agent.collection);
    res.json(docs);
  } catch (e) {
    console.error('List docs error:', e);
    res.status(500).json({ error: 'Failed to list docs' });
  }
});

app.delete('/docs/:agentId/:id', async (req, res) => {
  const agent = agents[req.params.agentId];
  if (!agent) return res.status(404).json({ error: 'Agent not found' });
  const idParam = req.params.id;
  const id = Number(idParam);
  const ok = await removeDoc(agent.collection, id);
  if (ok) res.json({ status: 'ok' });
  else res.status(500).json({ error: 'Delete failed' });
});

app.get('/chat/:id', (req, res) => {
  const agent = agents[req.params.id];
  if (!agent) return res.status(404).send('Agent not found');
  res.send(chatHtml(agent));
});

app.post('/chat/:id', async (req, res) => {
  console.log('CHAT POST', req.body);
  const agent = agents[req.params.id];
  if (!agent) return res.status(404).json({ error: 'Agent not found' });
  const { message } = req.body;
  const sessionId = req.ip + '-' + agent.id;
  try {
    const history = dashboardHistory[sessionId] || [];
    const context = await searchDocs(agent.collection, message, agent.topK);
    const answer = await askLLM(agent, context, message, history.concat({ role: 'user', text: message }));
    addDashboardMessage(sessionId, 'user', message);
    addDashboardMessage(sessionId, 'bot', answer);
    res.json({ answer });
  } catch (e) {
    console.error('Chat error:', e);
    res.status(500).json({ error: 'Failed to generate answer' });
  }
});

app.get('/user-history/:id', (req, res) => {
  res.send(userHistoryHtml(req.params.id));
});

app.get('/history', (req, res) => {
  res.send(historyHtml());
});

app.get('/', (req, res) => {
  res.send(homeHtml());
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log('Server running on port', PORT));

function getFirstAgent() {
  return Object.values(agents)[0];
}

// --- Telegram Bot Integration ---
const TelegramBot = require('node-telegram-bot-api');
const telegramBots = {};

function startTelegramBot(agent) {
  const token = agent.telegramToken;
  if (!token) return;

  if (telegramBots[agent.id]) {
    if (telegramBots[agent.id].token === token) return;
    telegramBots[agent.id].bot.stopPolling().catch((e) => console.error('Error stopping Telegram bot:', e));
  }

  const bot = new TelegramBot(token, { polling: { autoStart: false } });
  telegramBots[agent.id] = { bot, token };

  async function start(attempt = 0) {
    try {
      await bot.deleteWebHook({ drop_pending_updates: true });
      await bot.startPolling();
      console.log(`Telegram bot started for agent ${agent.id}`);
    } catch (e) {
      console.error('Failed to start Telegram bot:', e);
      if (attempt < 5 && (e?.response?.statusCode === 409 || String(e).includes('409'))) {
        const delay = 5000;
        console.log(`Retrying Telegram bot start in ${delay / 1000}s...`);
        setTimeout(() => start(attempt + 1), delay);
      }
    }
  }

  start();

  bot.on('polling_error', (error) => {
    console.error('Polling error:', error);
    if (error?.response?.statusCode === 409 || String(error).includes('409')) {
      bot.stopPolling()
        .then(() => bot.deleteWebHook({ drop_pending_updates: true }))
        .then(() => bot.startPolling())
        .catch((e) => console.error('Failed to restart polling:', e));
    }
  });

  bot.on('message', async (msg) => {
    const chatId = msg.chat.id;
    const text = msg.text?.trim();
    if (!text) return;
    try {
      const history = (telegramHistory[agent.id] && telegramHistory[agent.id][chatId]) || [];
      const context = await searchDocs(agent.collection, text, agent.topK);
      const answer = await askLLM(agent, context, text, history.concat({ role: 'user', text }).slice(-5));
      addTelegramMessage(agent.id, chatId, 'user', text);
      addTelegramMessage(agent.id, chatId, 'bot', answer);
      await bot.sendMessage(chatId, answer);
    } catch (e) {
      console.error('Telegram bot error:', e);
      addTelegramMessage(agent.id, chatId, 'bot', 'Failed to generate answer');
      await bot.sendMessage(chatId, 'Failed to generate answer');
    }
  });
}

function stopTelegramBot(agentId) {
  const entry = telegramBots[agentId];
  if (entry) {
    entry.bot.stopPolling().catch((e) => console.error('Error stopping Telegram bot:', e));
    delete telegramBots[agentId];
  }
}

Object.values(agents).forEach(startTelegramBot);

function shutdown() {
  Promise.all(Object.values(telegramBots).map((b) => b.bot.stopPolling().catch(() => {}))).finally(() => process.exit());
}

process.once('SIGINT', shutdown);
process.once('SIGTERM', shutdown);

// --- Viber Bot Integration ---
if (process.env.VIBER_AUTH_TOKEN) {
  const VIBER_API = 'https://chatapi.viber.com/pa';

  app.post('/viber/webhook', async (req, res) => {
    const { event, message, sender } = req.body;
    if (event === 'message' && message && sender && message.text) {
      const text = message.text.trim();
      const agent = getFirstAgent();
      if (!agent) return res.sendStatus(200);
      try {
        const context = await searchDocs(agent.collection, text, agent.topK);
        const answer = await askLLM(agent, context, text);
        await axios.post(`${VIBER_API}/send_message`, {
          receiver: sender.id,
          type: 'text',
          text: answer,
        }, { headers: { 'X-Viber-Auth-Token': process.env.VIBER_AUTH_TOKEN } });
      } catch (e) {
        console.error('Viber bot error:', e.response ? e.response.data : e);
        await axios.post(`${VIBER_API}/send_message`, {
          receiver: sender.id,
          type: 'text',
          text: 'Failed to generate answer',
        }, { headers: { 'X-Viber-Auth-Token': process.env.VIBER_AUTH_TOKEN } });
      }
    }
    res.sendStatus(200);
  });

  async function setViberWebhook() {
    if (!process.env.VIBER_WEBHOOK_URL) return;
    try {
      await axios.post(`${VIBER_API}/set_webhook`, {
        url: `${process.env.VIBER_WEBHOOK_URL}/viber/webhook`,
      }, { headers: { 'X-Viber-Auth-Token': process.env.VIBER_AUTH_TOKEN } });
      console.log('Viber webhook set');
    } catch (e) {
      console.error('Failed to set Viber webhook:', e.response ? e.response.data : e);
    }
  }

  setViberWebhook();
  console.log('Viber bot started');
} else {
  console.log('VIBER_AUTH_TOKEN not set, skipping Viber bot startup');
}
// --- End of Viber Bot Integration ---
// --- End of Telegram Bot Integration ---
