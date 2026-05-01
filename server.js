import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import { GoogleGenAI } from "@google/genai";

const app = express();
const port = Number(process.env.PORT || 3000);
const host = process.env.HOST || "127.0.0.1";
const model = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const botName = process.env.BOT_NAME || "Site Assistant";
const botInstructions =
  process.env.BOT_INSTRUCTIONS ||
  "You are a helpful website assistant. Keep answers concise, friendly, and accurate.";

const allowedOrigins = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

let ai;

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
      if (!origin || allowedOrigins.length === 0 || allowedOrigins.includes(origin)) {
        callback(null, true);
        return;
      }

      callback(new Error("Origin not allowed by ALLOWED_ORIGINS"));
    }
  })
);

app.get("/api/config", (_req, res) => {
  res.json({
    botName,
    model
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

    const contents = [
      {
        role: "user",
        parts: [{ text: buildPrompt(messages) }]
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
      reply: response.text || "I could not generate a response. Please try again."
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

function buildPrompt(messages) {
  const transcript = messages
    .map((message) => `${message.role === "assistant" ? "Assistant" : "User"}: ${message.content}`)
    .join("\n");

  return `${botInstructions}

Conversation:
${transcript}

Reply as ${botName}.`;
}
