import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import { GoogleGenAI } from "@google/genai";
import * as cheerio from "cheerio";

const siteUrl = normalizeUrl(process.env.SITE_URL || process.argv[2] || "");
const outputPath = process.env.RAG_INDEX_PATH || "data/knowledge-base.json";
const embeddingModel = process.env.EMBEDDING_MODEL || "gemini-embedding-001";
const maxPages = Number(process.env.MAX_CRAWL_PAGES || 40);
const chunkSize = Number(process.env.RAG_CHUNK_SIZE || 1400);
const chunkOverlap = Number(process.env.RAG_CHUNK_OVERLAP || 180);

if (!process.env.GEMINI_API_KEY || !siteUrl) {
  await writeEmptyIndex(
    !process.env.GEMINI_API_KEY
      ? "Missing GEMINI_API_KEY."
      : "Missing SITE_URL. Example: SITE_URL=https://www.aibuzzer.buzz npm run index"
  );
  process.exit(0);
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const urls = await discoverUrls(siteUrl);
const pages = [];

for (const url of urls.slice(0, maxPages)) {
  try {
    const page = await fetchPageText(url);

    if (page.text.length >= 240) {
      pages.push(page);
      console.log(`Indexed source: ${url}`);
    }
  } catch (error) {
    console.warn(`Skipped ${url}: ${error.message}`);
  }
}

const chunks = pages.flatMap((page) => chunkPage(page, chunkSize, chunkOverlap));

if (chunks.length === 0) {
  await writeEmptyIndex("No readable page content found. Check SITE_URL or sitemap access.");
  process.exit(0);
}

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

const index = {
  siteUrl,
  embeddingModel,
  outputDimensionality: 768,
  generatedAt: new Date().toISOString(),
  pageCount: pages.length,
  chunkCount: embeddedChunks.length,
  chunks: embeddedChunks
};

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, `${JSON.stringify(index)}\n`);

console.log(`Wrote ${embeddedChunks.length} chunks from ${pages.length} pages to ${outputPath}`);

async function writeEmptyIndex(reason) {
  const index = {
    siteUrl: siteUrl || null,
    embeddingModel,
    outputDimensionality: 768,
    generatedAt: new Date().toISOString(),
    pageCount: 0,
    chunkCount: 0,
    warning: reason,
    chunks: []
  };

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(index)}\n`);
  console.warn(`Wrote empty knowledge index: ${reason}`);
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
      "User-Agent": "website-chatbot-widget-indexer/0.1"
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

function chunkPage(page, size, overlap) {
  const chunks = [];
  let start = 0;

  while (start < page.text.length) {
    const text = page.text.slice(start, start + size).trim();

    if (text.length >= 180) {
      chunks.push({
        id: `${page.url}#chunk-${chunks.length + 1}`,
        url: page.url,
        title: page.title,
        text
      });
    }

    start += size - overlap;
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

function normalizeUrl(value) {
  if (!value) {
    return "";
  }

  return new URL(value).href.replace(/\/$/, "");
}

function decodeXml(value) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}
