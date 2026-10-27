/* Experiment 3 — TinyLlama chat UI */
(function () {
  "use strict";
  const { postJSON, request, setStatus, startBusy } = window.LabSpace;
  const form = document.getElementById("qa-form");
  if (!form) return;
  const chat = document.getElementById("chat");
  const empty = document.getElementById("chat-empty");
  const input = document.getElementById("qa-input");
  const send = document.getElementById("qa-send");
  const clear = document.getElementById("qa-clear");
  const deviceName = document.getElementById("qa-device-name");
  const stateEl = document.getElementById("qa-state");

  function addMessage(kind, text, meta) {
    if (empty) empty.hidden = true;
    const div = document.createElement("div");
    div.className = "msg " + kind;
    div.textContent = text;
    if (meta) {
      const m = document.createElement("span");
      m.className = "msg-meta";
      m.textContent = meta;
      div.appendChild(m);
    }
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
  }

  async function refreshStatus() {
    const res = await request("/api/exp3/status", {}, 15000);
    if (res.ok) {
      if (deviceName) deviceName.textContent = res.data.device;
      if (stateEl) stateEl.textContent = String(res.data.state).replace("_", " ");
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) {
      addMessage("msg-error", "Please enter a question.");
      input.focus();
      return;
    }
    addMessage("msg-user", question);
    input.value = "";
    send.disabled = true;

    const engineEl = document.querySelector('input[name="qa-engine"]:checked');
    const engine = engineEl ? engineEl.value : "local";

    const pending = addMessage("msg-bot", "");
    const stop = startBusy(pending, (engine === "gemini" ? "Asking Gemini…" : "Generating answer… (first question also loads the model)"));
    pending.setAttribute("aria-busy", "true");

    const res = await postJSON("/api/exp3/ask", {
      question: question,
      engine: engine,
      max_new_tokens: Number(document.getElementById("qa-max").value),
      temperature: Number(document.getElementById("qa-temp").value),
      top_p: Number(document.getElementById("qa-topp").value),
    }, 15 * 60 * 1000);

    stop();
    pending.removeAttribute("aria-busy");
    pending.className = "msg";
    pending.textContent = "";
    if (res.ok) {
      pending.classList.add("msg-bot");
      pending.textContent = res.data.answer;
      const meta = document.createElement("span");
      meta.className = "msg-meta";
      meta.textContent = (res.data.engine === "gemini" ? "Answered by Gemini (cloud, not TinyLlama) · " : "TinyLlama (local) · ") + res.data.model + " · " + res.data.device + " · " + res.data.seconds + "s";
      pending.appendChild(meta);
    } else {
      pending.classList.add("msg-error");
      pending.textContent = res.error;
    }
    chat.scrollTop = chat.scrollHeight;
    send.disabled = false;
    input.focus();
    refreshStatus();
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); form.requestSubmit(); }
  });

  clear.addEventListener("click", () => {
    chat.querySelectorAll(".msg").forEach((n) => n.remove());
    if (empty) empty.hidden = false;
  });
})();
