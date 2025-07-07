const { QdrantClient } = require('@qdrant/js-client-rest');
const fs = require('fs');

const client = new QdrantClient({ url: 'http://localhost:6333' });
const collectionName = "docs";

// Example: Create collection
async function setup() {
    try {
        await client.createCollection(collectionName, {
            vectors: {
                size: 384, // e.g., for MiniLM embeddings
                distance: "Cosine"
            }
        });
        console.log("Collection created!");
    } catch (e) {
        console.log("Collection might exist. Skipping create.");
    }
}

// Example: Ingest a simple text file (using random embeddings for now)
async function ingest() {
    await setup();
    const text = fs.readFileSync('mydoc.txt', 'utf-8');
    // For real RAG, use embeddings from LangChainJS, e.g. OpenAIEmbeddings
    const fakeEmbedding = Array(384).fill(Math.random()); // Replace with real embeddings!
    await client.upsert(collectionName, {
        points: [{
            id: 1,
            vector: fakeEmbedding,
            payload: { text }
        }]
    });
    console.log("Ingested document!");
}

ingest();
