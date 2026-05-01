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
- `EMBEDDING_MODEL`: Defaults to `gemini-embedding-001`.
- `HOST`: Defaults to `127.0.0.1`.
- `BOT_NAME`: Display name and reply persona.
- `BOT_INSTRUCTIONS`: System-style guidance added before each Gemini request.
- `ALLOWED_ORIGINS`: Comma-separated list of allowed website origins. Leave empty during local development only.
- `SITE_URL`: Website to crawl for RAG context, for example `https://www.aibuzzer.buzz`.
- `RAG_INDEX_PATH`: Defaults to `data/knowledge-base.json`.
- `MAX_CRAWL_PAGES`: Defaults to `40`.

## Notes

## Website Knowledge / RAG

Build the website knowledge index before deploying:

```bash
SITE_URL=https://www.aibuzzer.buzz npm run index
```

This crawls the site, creates Gemini embeddings, and writes `data/knowledge-base.json`.

On Vercel, set these environment variables in the chatbot project:

```bash
GEMINI_API_KEY=your_google_ai_studio_api_key
SITE_URL=https://www.aibuzzer.buzz
ALLOWED_ORIGINS=https://www.aibuzzer.buzz,https://aibuzzer.buzz
```

Then set the Vercel build command to:

```bash
npm run build
```

And the install command to:

```bash
npm install && npm run index
```

The chat API retrieves the most relevant indexed snippets and sends them to Gemini with each user question.

Check deployed index status:

```bash
curl https://your-chatbot-domain.vercel.app/api/knowledge
```

If `enabled` is `false`, check the latest Vercel deployment logs for the `npm run index`
step. The function bundle must include `data/knowledge-base.json`.
