import "dotenv/config";
import fs from "node:fs";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import { GoogleGenAI } from "@google/genai";
import * as cheerio from "cheerio";

const app = express();
const port = Number(process.env.PORT || 3000);
const host = process.env.HOST || "127.0.0.1";
const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const embeddingModel = process.env.EMBEDDING_MODEL || "gemini-embedding-001";
const ragIndexPath = process.env.RAG_INDEX_PATH || "data/knowledge-base.json";
const siteUrl = normalizeSiteUrl(process.env.SITE_URL || "");
const maxCrawlPages = Number(process.env.MAX_CRAWL_PAGES || 12);
const ragChunkSize = Number(process.env.RAG_CHUNK_SIZE || 1400);
const ragChunkOverlap = Number(process.env.RAG_CHUNK_OVERLAP || 180);
const botName = process.env.BOT_NAME || "Site Assistant";
const botInstructions =
  process.env.BOT_INSTRUCTIONS ||
  "You are a helpful website assistant. Keep answers concise, friendly, and accurate.";

const allowedOrigins = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => normalizeOrigin(origin))
  .filter(Boolean);
const trackedSources = [
  {
    name: "OpenAI",
    siteUrl: "https://openai.com/newsroom/",
    description: "Models, APIs, ChatGPT, safety, platform launches",
    latest: "Apr 30, 2026, 5:30 AM"
  },
  {
    name: "Anthropic",
    siteUrl: "https://www.anthropic.com/news",
    description: "Claude models, enterprise launches, policy, safety",
    latest: "Apr 28, 2026, 5:30 AM"
  },
  {
    name: "Google AI",
    siteUrl: "https://blog.google/technology/ai/",
    description: "Gemini, AI Mode, Workspace AI, developer tooling",
    latest: "Apr 28, 2026, 9:30 PM"
  },
  {
    name: "Google DeepMind",
    siteUrl: "https://deepmind.google/",
    description: "Research, multimodal models, robotics, science",
    latest: "Feb 1, 2026, 5:30 AM"
  },
  {
    name: "Meta AI",
    siteUrl: "https://ai.meta.com/blog/",
    description: "Llama, research, open models, consumer AI",
    latest: "Apr 8, 2026, 5:30 AM"
  },
  {
    name: "Mistral AI",
    siteUrl: "https://mistral.ai/news",
    description: "Frontier models, enterprise releases, open weights",
    latest: "Apr 30, 2026, 3:26 AM"
  },
  {
    name: "Microsoft Research",
    siteUrl: "https://www.microsoft.com/en-us/research/blog/",
    description: "AI research, language models, agents, cloud AI, enterprise",
    latest: "May 1, 2026, 3:23 AM"
  },
  {
    name: "Hugging Face",
    siteUrl: "https://huggingface.co/blog",
    description: "Open-source releases, tooling, research, community",
    latest: "Apr 29, 2026, 10:15 PM"
  }
];

let ai;
let knowledgeBase;
let knowledgeBasePromise;
const liveSourceCache = new Map();

app.use(
  helmet({
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false
  })
);
app.use(express.json({ limit: "1mb" }));
app.use(express.static("public"));

app.use(
  cors({
    origin(origin, callback) {
      const normalizedOrigin = normalizeOrigin(origin);

      if (
        !normalizedOrigin ||
        allowedOrigins.length === 0 ||
        allowedOrigins.includes(normalizedOrigin)
      ) {
        callback(null, true);
        return;
      }

      callback(null, false);
    }
  })
);

app.get("/api/config", (_req, res) => {
  res.json({
    botName,
    model
  });
});

app.get("/api/knowledge", async (_req, res) => {
  const index = await getKnowledgeBase();

  res.json({
    enabled: Boolean(index?.chunks?.length),
    path: ragIndexPath,
    fileExists: fs.existsSync(ragIndexPath),
    siteUrl: index?.siteUrl || null,
    generatedAt: index?.generatedAt || null,
    pageCount: index?.pageCount || 0,
    chunkCount: index?.chunkCount || 0,
    warning: index?.warning || null
  });
});

app.post("/api/chat", async (req, res) => {
  try {
    if (!process.env.GEMINI_API_KEY) {
      res.status(500).json({
        error: "Missing GEMINI_API_KEY. Add it to .env and restart the server."
      });
      return;
    }

    ai ||= new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

    const messages = normalizeMessages(req.body?.messages);
    const latestUserMessage = messages.at(-1)?.content;

    if (!latestUserMessage) {
      res.status(400).json({ error: "Send at least one user message." });
      return;
    }

    const context = await retrieveContext(latestUserMessage);
    const contents = [
      {
        role: "user",
        parts: [{ text: buildPrompt(messages, context) }]
      }
    ];

    const response = await ai.models.generateContent({
      model,
      contents,
      config: {
        temperature: 0.4,
        maxOutputTokens: 800
      }
    });

    res.json({
      reply: response.text || "I could not generate a response. Please try again.",
      sources: context.map((item) => ({
        title: item.title,
        url: item.url
      }))
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: "The assistant could not respond right now."
    });
  }
});

if (!process.env.VERCEL) {
  app.listen(port, host, () => {
    console.log(`Gemini chatbot server running at http://${host}:${port}`);
  });
}

export default app;

function normalizeMessages(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .slice(-12)
    .map((message) => ({
      role: message?.role === "assistant" ? "assistant" : "user",
      content: String(message?.content || "").trim().slice(0, 4000)
    }))
    .filter((message) => message.content);
}

async function retrieveContext(query) {
  const index = await getKnowledgeBase();

  if (!index?.chunks?.length) {
    return [];
  }

  const response = await ai.models.embedContent({
    model: index.embeddingModel || embeddingModel,
    contents: query,
    config: { outputDimensionality: index.outputDimensionality || 768 }
  });
  const queryEmbedding = response.embeddings?.[0]?.values;

  if (!queryEmbedding) {
    return [];
  }

  const queryTerms = extractTerms(query);

  const semanticChunks = index.chunks
    .map((chunk) => {
      const semanticScore = cosineSimilarity(queryEmbedding, chunk.embedding);
      const keywordScore = keywordMatchScore(queryTerms, `${chunk.title} ${chunk.text}`);

      return {
        ...chunk,
        score: semanticScore + keywordScore
      };
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, 8)
    .filter((chunk) => chunk.score > 0.35);
  const liveSourceChunks = await fetchMentionedSourceChunks(query);

  return [...liveSourceChunks, ...semanticChunks].slice(0, 10);
}

async function getKnowledgeBase() {
  if (!process.env.GEMINI_API_KEY) {
    return null;
  }

  ai ||= new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

  if (knowledgeBase !== undefined) {
    return knowledgeBase;
  }

  try {
    knowledgeBase = JSON.parse(fs.readFileSync(ragIndexPath, "utf8"));
    return knowledgeBase;
  } catch {}

  if (!siteUrl) {
    knowledgeBase = {
      siteUrl: null,
      generatedAt: new Date().toISOString(),
      pageCount: 0,
      chunkCount: 0,
      warning: "Missing SITE_URL.",
      chunks: []
    };
    return knowledgeBase;
  }

  knowledgeBasePromise ||= buildRuntimeKnowledgeBase();

  try {
    knowledgeBase = await knowledgeBasePromise;
  } catch (error) {
    console.error(error);
    knowledgeBase = {
      siteUrl,
      generatedAt: new Date().toISOString(),
      pageCount: 0,
      chunkCount: 0,
      warning: error.message,
      chunks: []
    };
  }

  return knowledgeBase;
}

async function buildRuntimeKnowledgeBase() {
  const urls = await discoverUrls(siteUrl);
  const pages = [];

  for (const url of urls.slice(0, maxCrawlPages)) {
    try {
      const page = await fetchPageText(url);

      if (page.text.length >= 240) {
        pages.push(page);
      }
    } catch (error) {
      console.warn(`Skipped ${url}: ${error.message}`);
    }
  }

  const apiChunks = await fetchAibuzzerApiChunks();
  const chunks = [
    ...buildTrackedSourceChunks(),
    ...apiChunks,
    ...pages.flatMap((page) => chunkPage(page))
  ];
  const embeddedChunks = [];

  for (const batch of batchItems(chunks, 20)) {
    const response = await ai.models.embedContent({
      model: embeddingModel,
      contents: batch.map((chunk) => chunk.text),
      config: { outputDimensionality: 768 }
    });

    response.embeddings.forEach((embedding, index) => {
      embeddedChunks.push({
        ...batch[index],
        embedding: embedding.values
      });
    });
  }

  return {
    siteUrl,
    embeddingModel,
    outputDimensionality: 768,
    generatedAt: new Date().toISOString(),
    pageCount: pages.length,
    chunkCount: embeddedChunks.length,
    chunks: embeddedChunks,
    warning: embeddedChunks.length ? null : "No readable website content was indexed."
  };
}

async function fetchAibuzzerApiChunks() {
  if (!siteUrl) {
    return [];
  }

  try {
    const response = await fetch(new URL("/api/updates", siteUrl), {
      headers: {
        "User-Agent": "website-chatbot-widget/0.1"
      }
    });

    if (!response.ok) {
      return [];
    }

    const snapshot = await response.json();
    const chunks = [];

    if (Array.isArray(snapshot.updates)) {
      chunks.push(
        ...snapshot.updates.map((update) => ({
          id: `aibuzzer-update-${slugify(update.id || update.title)}`,
          url: update.url || siteUrl,
          title: `${update.sourceName || "AIBuzzer"}: ${update.title}`,
          text: [
            `Source: ${update.sourceName}`,
            `Category: ${update.category}`,
            `Impact area: ${update.impactArea}`,
            `Published: ${formatIsoDate(update.publishedAt)}`,
            `Title: ${update.title}`,
            `Summary: ${decodeHtml(update.excerpt || "")}`,
            `Impact: ${decodeHtml(update.impact || "")}`,
            `URL: ${update.url}`
          ]
            .filter(Boolean)
            .join("\n")
        }))
      );
    }

    if (Array.isArray(snapshot.sourceStatuses)) {
      chunks.push(
        ...snapshot.sourceStatuses.map((source) => ({
          id: `aibuzzer-source-${slugify(source.id || source.name)}`,
          url: source.siteUrl || siteUrl,
          title: `${source.name} source health`,
          text: [
            `Source: ${source.name}`,
            `Focus: ${source.focus}`,
            `Official channel: ${source.siteUrl}`,
            `Status: ${source.ok ? "healthy" : "unavailable"}`,
            `Recent items available: ${source.itemCount}`,
            `Latest item: ${formatIsoDate(source.latestPublishedAt)}`
          ]
            .filter(Boolean)
            .join("\n")
        }))
      );
    }

    if (Array.isArray(snapshot.briefing)) {
      chunks.push({
        id: "aibuzzer-executive-briefing",
        url: siteUrl,
        title: "AIBuzzer executive briefing",
        text: snapshot.briefing.map((item, index) => `${index + 1}. ${item}`).join("\n")
      });
    }

    return chunks;
  } catch (error) {
    console.warn(`Could not fetch AIBuzzer API updates: ${error.message}`);
    return [];
  }
}

function buildPrompt(messages, context = []) {
  const transcript = messages
    .map((message) => `${message.role === "assistant" ? "Assistant" : "User"}: ${message.content}`)
    .join("\n");
  const sourceContext = context.length
    ? context
        .map(
          (item, index) =>
            `[${index + 1}] ${item.title}\nURL: ${item.url}\n${item.text}`
        )
        .join("\n\n")
    : "No indexed website context was found for this question.";
  const trackedSourceContext = trackedSources
    .map(
      (source) =>
        `- ${source.name}: ${source.description}. 6 recent items available. Latest: ${source.latest}.`
    )
    .join("\n");

  return `${botInstructions}

Use the indexed website context below when it is relevant. If the context does not contain the answer, say that the indexed site content does not include that detail yet.

AIBuzzer tracked official source channels:
${trackedSourceContext}

Indexed website context:
${sourceContext}

Conversation:
${transcript}

Reply as ${botName}.`;
}

function cosineSimilarity(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return 0;
  }

  let dot = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;

  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftMagnitude += left[index] * left[index];
    rightMagnitude += right[index] * right[index];
  }

  return dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude) || 1);
}

function buildTrackedSourceChunks() {
  return [
    {
      id: `${siteUrl || "aibuzzer"}#tracked-sources`,
      url: siteUrl || "",
      title: "AIBuzzer tracked official source channels",
      text: `AIBuzzer tracks these official channels: ${trackedSources
        .map(
          (source) =>
            `${source.name} (${source.description}; official source ${source.siteUrl}; 6 recent items available; latest ${source.latest})`
        )
        .join("; ")}.`
    },
    ...trackedSources.map((source) => ({
      id: `${siteUrl || "aibuzzer"}#source-${slugify(source.name)}`,
      url: source.siteUrl,
      title: `${source.name} source health`,
      text: `${source.name}: ${source.description}. AIBuzzer tracks this official channel at ${source.siteUrl}. 6 recent items available. Latest item: ${source.latest}.`
    }))
  ];
}

async function fetchMentionedSourceChunks(query) {
  const queryText = String(query || "").toLowerCase();
  const matches = trackedSources.filter((source) => {
    const sourceTerms = [source.name, ...source.name.split(/\s+/), slugify(source.name)]
      .map((term) => term.toLowerCase())
      .filter((term) => term.length > 2);

    return sourceTerms.some((term) => queryText.includes(term));
  });

  const chunks = [];

  for (const source of matches.slice(0, 2)) {
    chunks.push(...(await fetchLiveSourceChunks(source)));
  }

  return chunks;
}

async function fetchLiveSourceChunks(source) {
  const cacheKey = source.name;

  if (liveSourceCache.has(cacheKey)) {
    return liveSourceCache.get(cacheKey);
  }

  try {
    const html = await fetchText(source.siteUrl);
    const items = extractSourceItems(html, source.siteUrl).slice(0, 8);
    const text = items.length
      ? `${source.name} official source items:\n${items
          .map((item, index) => `${index + 1}. ${item.title} - ${item.url}`)
          .join("\n")}`
      : `${source.name} official source page: ${cleanText(cheerio.load(html)("body").text()).slice(0, 2400)}`;
    const chunks = [
      {
        id: `live-source-${slugify(source.name)}`,
        url: source.siteUrl,
        title: `${source.name} official source updates`,
        text,
        score: 1.2
      }
    ];

    liveSourceCache.set(cacheKey, chunks);
    return chunks;
  } catch (error) {
    console.warn(`Could not fetch ${source.name} source page: ${error.message}`);
    return [];
  }
}

function extractSourceItems(html, baseUrl) {
  const $ = cheerio.load(html);
  const items = [];
  const seen = new Set();

  $("a[href]").each((_, element) => {
    const title = cleanText($(element).text());

    if (title.length < 8) {
      return;
    }

    addSourceItem(items, seen, title, $(element).attr("href"), baseUrl);
  });

  for (const match of html.matchAll(/"value":"(https?:\\?\/\\?\/[^"]+)","label":"Read\s+([^"]+)"/g)) {
    addSourceItem(items, seen, decodeJsonish(match[2]), decodeJsonish(match[1]), baseUrl);
  }

  for (const match of html.matchAll(/"label":"Read\s+([^"]+)","page_location":"[^"]+"/g)) {
    addSourceItem(items, seen, decodeJsonish(match[1]), baseUrl, baseUrl);
  }

  return items;
}

function addSourceItem(items, seen, title, href, baseUrl) {
  try {
    const url = new URL(decodeJsonish(href), baseUrl);
    const cleanTitle = cleanText(decodeJsonish(title).replace(/^Read\s+/i, ""));
    const key = `${cleanTitle}:${url.href}`;

    if (!cleanTitle || seen.has(key)) {
      return;
    }

    seen.add(key);
    items.push({
      title: cleanTitle,
      url: url.href
    });
  } catch {}
}

function extractTerms(value) {
  return new Set(
    String(value || "")
      .toLowerCase()
      .match(/[a-z0-9]+/g)
      ?.filter((term) => term.length > 2) || []
  );
}

function keywordMatchScore(queryTerms, value) {
  const haystack = String(value || "").toLowerCase();
  let score = 0;

  for (const term of queryTerms) {
    if (haystack.includes(term)) {
      score += 0.12;
    }
  }

  return Math.min(score, 0.48);
}

async function discoverUrls(rootUrl) {
  const sitemapUrls = await readSitemap(rootUrl);

  if (sitemapUrls.length > 0) {
    return uniqueUrls([rootUrl, ...sitemapUrls], rootUrl);
  }

  const html = await fetchText(rootUrl);
  const $ = cheerio.load(html);
  const links = $("a[href]")
    .map((_, element) => new URL($(element).attr("href"), rootUrl).href)
    .get();

  return uniqueUrls([rootUrl, ...links], rootUrl);
}

async function readSitemap(rootUrl) {
  try {
    const sitemapUrl = new URL("/sitemap.xml", rootUrl).href;
    const xml = await fetchText(sitemapUrl);
    const matches = [...xml.matchAll(/<loc>(.*?)<\/loc>/gims)];
    return matches.map((match) => decodeXml(match[1].trim()));
  } catch {
    return [];
  }
}

async function fetchPageText(url) {
  const html = await fetchText(url);
  const $ = cheerio.load(html);

  $("script, style, noscript, svg, canvas, iframe, nav, footer, form").remove();

  const title = cleanText($("title").first().text());
  const description = cleanText($('meta[name="description"]').attr("content") || "");
  const headings = $("h1, h2, h3")
    .map((_, element) => cleanText($(element).text()))
    .get()
    .filter(Boolean)
    .join("\n");
  const body = cleanText($("main").text() || $("body").text());
  const text = cleanText([title, description, headings, body].filter(Boolean).join("\n\n"));

  return { url, title: title || url, text };
}

async function fetchText(url) {
  const response = await fetch(url, {
    headers: {
      "User-Agent": "website-chatbot-widget/0.1"
    }
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  return response.text();
}

function uniqueUrls(urls, rootUrl) {
  const root = new URL(rootUrl);
  const seen = new Set();

  return urls
    .map((url) => {
      try {
        const parsed = new URL(url, rootUrl);
        parsed.hash = "";
        parsed.search = "";
        return parsed;
      } catch {
        return null;
      }
    })
    .filter((url) => url && url.origin === root.origin)
    .filter((url) => !/\.(png|jpe?g|gif|webp|svg|pdf|zip|mp4|mp3)$/i.test(url.pathname))
    .map((url) => url.href.replace(/\/$/, "") || rootUrl)
    .filter((url) => {
      if (seen.has(url)) {
        return false;
      }

      seen.add(url);
      return true;
    });
}

function chunkPage(page) {
  const chunks = [];
  let start = 0;

  while (start < page.text.length) {
    const text = page.text.slice(start, start + ragChunkSize).trim();

    if (text.length >= 180) {
      chunks.push({
        id: `${page.url}#chunk-${chunks.length + 1}`,
        url: page.url,
        title: page.title,
        text
      });
    }

    start += ragChunkSize - ragChunkOverlap;
  }

  return chunks;
}

function batchItems(items, size) {
  const batches = [];

  for (let index = 0; index < items.length; index += size) {
    batches.push(items.slice(index, index + size));
  }

  return batches;
}

function cleanText(value) {
  return String(value || "")
    .replace(/\s+/g, " ")
    .replace(/\s+([.,!?;:])/g, "$1")
    .trim();
}

function normalizeOrigin(origin) {
  return String(origin || "")
    .trim()
    .replace(/\/+$/, "");
}

function normalizeSiteUrl(value) {
  if (!value) {
    return "";
  }

  try {
    return new URL(value).href.replace(/\/$/, "");
  } catch {
    return "";
  }
}

function slugify(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function formatIsoDate(value) {
  if (!value) {
    return "";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "Asia/Kolkata"
  });
}

function decodeHtml(value) {
  return String(value || "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&#39;/g, "'");
}

function decodeJsonish(value) {
  return decodeHtml(String(value || "").replace(/\\\//g, "/").replace(/\\"/g, '"'));
}

function decodeXml(value) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}
