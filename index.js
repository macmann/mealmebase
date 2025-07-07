require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const { QdrantClient } = require('@qdrant/js-client-rest');

const app = express();
app.use(bodyParser.json());

// Facebook verify webhook
app.get('/webhook', (req, res) => {
    if (req.query['hub.verify_token'] === process.env.VERIFY_TOKEN) {
        res.send(req.query['hub.challenge']);
    } else {
        res.send('Error, wrong token');
    }
});

// Receive messages
app.post('/webhook', async (req, res) => {
    const entries = req.body.entry;
    if (entries) {
        for (let entry of entries) {
            for (let event of entry.messaging) {
                if (event.message && event.message.text) {
                    const senderId = event.sender.id;
                    const messageText = event.message.text;
                    // Here: process messageText with RAG and reply
                    await handleMessage(senderId, messageText);
                }
            }
        }
    }
    res.sendStatus(200);
});

// Messenger reply
async function sendMessage(senderId, text) {
    await axios.post(
        `https://graph.facebook.com/v12.0/me/messages?access_token=${process.env.PAGE_ACCESS_TOKEN}`,
        {
            recipient: { id: senderId },
            message: { text }
        }
    );
}

// RAG (dummy, fill this in later)
async function handleMessage(senderId, messageText) {
    const { OpenAIEmbeddings } = require('@langchain/openai');
    const { QdrantClient } = require('@qdrant/js-client-rest');

    const client = new QdrantClient({ url: process.env.QDRANT_URL });
    const embeddings = new OpenAIEmbeddings({ apiKey: 'your_openai_api_key' });

    async function handleMessage(senderId, messageText) {
        // 1. Generate embedding for the question
        const queryEmbedding = await embeddings.embedQuery(messageText);

        // 2. Search in Qdrant
        const searchResult = await client.search('docs', {
            vector: queryEmbedding,
            limit: 1,
            with_payload: true
        });

        let answer = "Sorry, I couldn't find an answer.";
        if (searchResult && searchResult.length > 0) {
            answer = searchResult[0].payload.text;
        }

        await sendMessage(senderId, answer);
    }

}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
