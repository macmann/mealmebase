# MealMeBase RAG Chatbot

This project is a simple RAG (Retrieval Augmented Generation) chatbot demo built with Node.js and Express.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file in the project root (or update the existing one) with the following entries:
   ```env
   PAGE_ACCESS_TOKEN=your_facebook_page_access_token
   VERIFY_TOKEN=your_own_verify_token
   QDRANT_URL=http://localhost:6333
   OPENAI_API_KEY=your_openai_api_key
   ```
   The `OPENAI_API_KEY` is required for generating embeddings and chatting with the OpenAI API.

3. Start the server:
   ```bash
   npm start
   ```

The server will run on port defined by the `PORT` environment variable or `3000` by default.
