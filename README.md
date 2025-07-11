# MealMeBase RAG Chatbot

This project is a simple RAG (Retrieval Augmented Generation) chatbot demo built with Node.js and Express. The frontend is styled using [Tailwind CSS](https://tailwindcss.com) for a clean and modern look.

The admin interface allows uploading knowledge sources used by the chatbot. You can now upload plain text, **CSV**, **PDF**, or **JSON** documents.

Chat panels render messages using **Markdown**, allowing rich formatting in both user and bot messages.

Telegram chat history from the bot is saved to `chathistory.json` and can be viewed in the "Chat History" page.

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
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   VIBER_AUTH_TOKEN=your_viber_auth_token
   VIBER_WEBHOOK_URL=https://your-domain.com
   LOGO_URL=https://ibb.co/HDy3fYZ8
   ```
   You can copy `.env.example` as a starting point and fill in your keys.
   The `OPENAI_API_KEY` is required for generating embeddings and chatting with the OpenAI API.
   The `VIBER_AUTH_TOKEN` and `VIBER_WEBHOOK_URL` are used to configure the Viber webhook integration.

3. Start the server:
   ```bash
   npm start
   ```

The server will run on port defined by the `PORT` environment variable or `3000` by default.
