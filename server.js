import "dotenv/config";
import fs from "node:fs";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import { GoogleGenAI } from "@google/genai";

const app = express();
const port = Number(process.env.PORT || 3000);
const host = process.env.HOST || "127.0.0.1";
const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const embeddingModel = process.env.EMBEDDING_MODEL || "gemini-embedding-001";
const ragIndexPath = process.env.RAG_INDEX_PATH || "data/knowledge-base.json";
const botName = process.env.BOT_NAME || "Site Assistant";
const botInstructions =
  process.env.BOT_INSTRUCTIONS ||
  "You are a helpful website assistant. Keep answers concise, friendly, and accurate.";

const allowedOrigins = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => normalizeOrigin(origin))
  .filter(Boolean);

let ai;
let knowledgeBase;

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

app.get("/api/knowledge", (_req, res) => {
  const index = loadKnowledgeBase();

  res.json({
    enabled: Boolean(index?.chunks?.length),
    path: ragIndexPath,
    fileExists: fs.existsSync(ragIndexPath),
    siteUrl: index?.siteUrl || null,
    generatedAt: index?.generatedAt || null,
    pageCount: index?.pageCount || 0,
    chunkCount: index?.chunkCount || 0
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
  const index = loadKnowledgeBase();

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

  return index.chunks
    .map((chunk) => ({
      ...chunk,
      score: cosineSimilarity(queryEmbedding, chunk.embedding)
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 5)
    .filter((chunk) => chunk.score > 0.45);
}

function loadKnowledgeBase() {
  if (knowledgeBase !== undefined) {
    return knowledgeBase;
  }

  try {
    knowledgeBase = JSON.parse(fs.readFileSync(ragIndexPath, "utf8"));
  } catch {
    knowledgeBase = null;
  }

  return knowledgeBase;
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

  return `${botInstructions}

Use the indexed website context below when it is relevant. If the context does not contain the answer, say that the indexed site content does not include that detail yet.

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

function normalizeOrigin(origin) {
  return String(origin || "")
    .trim()
    .replace(/\/+$/, "");
}
