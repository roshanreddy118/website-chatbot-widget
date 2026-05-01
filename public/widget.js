(function () {
  const currentScript = document.currentScript;
  const scriptUrl = new URL(currentScript.src);
  const apiBase = currentScript.dataset.apiUrl || scriptUrl.origin;
  const botName = currentScript.dataset.botName || "Site Assistant";
  const accent = currentScript.dataset.accent || "#2563eb";
  const welcome =
    currentScript.dataset.welcome || "Hi, I can help answer questions about this website.";

  const style = document.createElement("style");
  style.textContent = `
    .gcb-launcher {
      position: fixed;
      right: 20px;
      bottom: 20px;
      z-index: 2147483647;
      width: 56px;
      height: 56px;
      border: 0;
      border-radius: 50%;
      background: ${accent};
      color: #fff;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.24);
      cursor: pointer;
      font: 700 23px/1 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .gcb-panel {
      position: fixed;
      right: 20px;
      bottom: 88px;
      z-index: 2147483647;
      display: none;
      width: min(380px, calc(100vw - 32px));
      height: min(560px, calc(100vh - 120px));
      overflow: hidden;
      border: 1px solid #dbe3ef;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 22px 70px rgba(15, 23, 42, 0.28);
      color: #101828;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .gcb-panel[data-open="true"] {
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    .gcb-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      background: #f8fafc;
      border-bottom: 1px solid #e4eaf3;
      font-weight: 700;
    }

    .gcb-close {
      width: 32px;
      height: 32px;
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: #475467;
      cursor: pointer;
      font-size: 22px;
      line-height: 1;
    }

    .gcb-messages {
      overflow-y: auto;
      padding: 16px;
      background: #fbfdff;
    }

    .gcb-message {
      width: fit-content;
      max-width: 86%;
      margin: 0 0 12px;
      padding: 10px 12px;
      border-radius: 8px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 14px;
      line-height: 1.45;
    }

    .gcb-message[data-role="assistant"] {
      background: #eef4ff;
      color: #1d2939;
    }

    .gcb-message[data-role="user"] {
      margin-left: auto;
      background: ${accent};
      color: #fff;
    }

    .gcb-form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid #e4eaf3;
      background: #fff;
    }

    .gcb-input {
      min-width: 0;
      border: 1px solid #ccd5e1;
      border-radius: 8px;
      padding: 11px 12px;
      font: 14px/1.35 inherit;
      color: #101828;
    }

    .gcb-send {
      border: 0;
      border-radius: 8px;
      padding: 0 15px;
      background: ${accent};
      color: #fff;
      font: 700 14px/1 inherit;
      cursor: pointer;
    }

    .gcb-send:disabled,
    .gcb-input:disabled {
      opacity: 0.62;
      cursor: wait;
    }
  `;

  const panel = document.createElement("section");
  panel.className = "gcb-panel";
  panel.setAttribute("aria-label", `${botName} chat`);
  panel.innerHTML = `
    <div class="gcb-header">
      <span>${escapeHtml(botName)}</span>
      <button class="gcb-close" type="button" aria-label="Close chat">×</button>
    </div>
    <div class="gcb-messages" aria-live="polite"></div>
    <form class="gcb-form">
      <input class="gcb-input" name="message" type="text" autocomplete="off" placeholder="Ask a question" />
      <button class="gcb-send" type="submit">Send</button>
    </form>
  `;

  const launcher = document.createElement("button");
  launcher.className = "gcb-launcher";
  launcher.type = "button";
  launcher.setAttribute("aria-label", `Open ${botName} chat`);
  launcher.textContent = "?";

  document.head.appendChild(style);
  document.body.append(panel, launcher);

  const messagesEl = panel.querySelector(".gcb-messages");
  const form = panel.querySelector(".gcb-form");
  const input = panel.querySelector(".gcb-input");
  const send = panel.querySelector(".gcb-send");
  const close = panel.querySelector(".gcb-close");
  const messages = [{ role: "assistant", content: welcome }];

  renderMessages();

  launcher.addEventListener("click", () => {
    panel.dataset.open = panel.dataset.open === "true" ? "false" : "true";
    if (panel.dataset.open === "true") {
      input.focus();
    }
  });

  close.addEventListener("click", () => {
    panel.dataset.open = "false";
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const content = input.value.trim();

    if (!content) {
      return;
    }

    messages.push({ role: "user", content });
    input.value = "";
    setLoading(true);
    renderMessages("Thinking...");

    try {
      const response = await fetch(`${apiBase}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: messages.filter((message) => message.role !== "system") })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }

      messages.push({ role: "assistant", content: data.reply });
    } catch (error) {
      messages.push({
        role: "assistant",
        content: "Sorry, I could not connect to the assistant. Please try again in a moment."
      });
    } finally {
      setLoading(false);
      renderMessages();
    }
  });

  function renderMessages(temporaryText) {
    const visibleMessages = temporaryText
      ? [...messages, { role: "assistant", content: temporaryText }]
      : messages;

    messagesEl.innerHTML = visibleMessages
      .map(
        (message) =>
          `<div class="gcb-message" data-role="${message.role}">${escapeHtml(message.content)}</div>`
      )
      .join("");
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setLoading(value) {
    input.disabled = value;
    send.disabled = value;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
})();
