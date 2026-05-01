# Gemini Chatbot Embed

Embeddable chatbot starter for websites using Google AI Studio / Gemini. The browser widget is public, but Gemini calls go through the Node server so your API key is never exposed.

## Setup

1. Create a Gemini API key in Google AI Studio.
2. Copy `.env.example` to `.env`.
3. Set `GEMINI_API_KEY`.
4. Install and run:

```bash
npm install
npm run dev
```

Open `http://localhost:3000` to test the widget.

## Embed On Any Website

Deploy this server, then add this to the target website:

```html
<script
  src="https://your-domain.com/widget.js"
  data-api-url="https://your-domain.com"
  data-bot-name="Site Assistant"
  data-accent="#2563eb"
  data-welcome="Hi, I can help answer questions about this website."
></script>
```

Set `ALLOWED_ORIGINS` in `.env` to the sites allowed to call your API:

```bash
ALLOWED_ORIGINS=https://example.com,https://www.example.com
```

## Configuration

- `GEMINI_API_KEY`: Google AI Studio API key.
- `GEMINI_MODEL`: Defaults to `gemini-2.5-flash`.
- `HOST`: Defaults to `127.0.0.1`.
- `BOT_NAME`: Display name and reply persona.
- `BOT_INSTRUCTIONS`: System-style guidance added before each Gemini request.
- `ALLOWED_ORIGINS`: Comma-separated list of allowed website origins. Leave empty during local development only.

## Notes

This is an embeddable chat widget, not document retrieval yet. If you want the bot to answer from each website's pages or uploaded documents, the next step is adding crawling plus vector search/RAG.
