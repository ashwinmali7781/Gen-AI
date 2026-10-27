/* Experiment 9 — PDF question answering (RAG) */
(function () {
  "use strict";
  const { request, postJSON, setStatus, startBusy } = window.LabSpace;
  const upForm = document.getElementById("rag-upload-form");
  if (!upForm) return;

  const fileInput = document.getElementById("rag-file");
  const upBtn = document.getElementById("rag-upload-btn");
  const upStatus = document.getElementById("rag-upload-status");
  const docsEl = document.getElementById("rag-docs");
  const docsEmpty = document.getElementById("rag-docs-empty");
  const askForm = document.getElementById("rag-ask-form");
  const askBtn = document.getElementById("rag-ask-btn");
  const question = document.getElementById("rag-question");
  const status = document.getElementById("rag-status");
  const resultEl = document.getElementById("rag-result");
  const answerEl = document.getElementById("rag-answer");
  const noteEl = document.getElementById("rag-gen-note");
  const sourcesEl = document.getElementById("rag-sources");

  let docs = [];

  function renderDocs() {
    docsEl.textContent = "";
    docsEmpty.hidden = docs.length > 0;
    docs.forEach((d) => {
      const li = document.createElement("li");
      const label = document.createElement("span");
      const name = document.createElement("strong");
      name.textContent = d.name;
      const meta = document.createElement("span");
      meta.className = "doc-meta";
      meta.textContent = "  " + d.kind.toUpperCase() + " · " + d.pages + (d.pages === 1 ? " page" : " pages") + " · " + d.chunks + " chunks";
      label.appendChild(name);
      label.appendChild(meta);
      const del = document.createElement("button");
      del.type = "button";
      del.className = "btn btn-ghost btn-sm";
      del.textContent = "Remove";
      del.setAttribute("aria-label", "Remove " + d.name);
      del.addEventListener("click", async () => {
        del.disabled = true;
        const res = await request("/api/exp9/documents/" + encodeURIComponent(d.doc_id), { method: "DELETE" }, 30000);
        if (res.ok) { docs = res.data.documents; renderDocs(); }
        else { setStatus(upStatus, "error", res.error); del.disabled = false; }
      });
      li.appendChild(label);
      li.appendChild(del);
      docsEl.appendChild(li);
    });
  }

  try { docs = JSON.parse(docsEl.dataset.initial || "[]"); } catch (_) { docs = []; }
  renderDocs();

  upForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) { setStatus(upStatus, "error", "Choose a PDF or .txt file first."); return; }
    const lower = file.name.toLowerCase();
    if (!(lower.endsWith(".pdf") || lower.endsWith(".txt"))) {
      setStatus(upStatus, "error", "Unsupported file type. Upload a PDF or .txt file.");
      return;
    }
    upBtn.disabled = true;
    const stop = startBusy(upStatus, "Extracting, chunking and embedding (first upload loads the embedding model)…");
    const body = new FormData();
    body.append("file", file);
    const res = await request("/api/exp9/upload", { method: "POST", body: body }, 10 * 60 * 1000);
    stop();
    upBtn.disabled = false;
    if (!res.ok) { setStatus(upStatus, "error", res.error); return; }
    docs = res.data.documents;
    renderDocs();
    fileInput.value = "";
    setStatus(upStatus, "success", "Indexed “" + res.data.document.name + "” into " + res.data.document.chunks + " chunks.");
  });

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = question.value.trim();
    if (!q) { setStatus(status, "error", "Please enter a question."); question.focus(); return; }
    if (!docs.length) { setStatus(status, "error", "Upload and index a document first."); return; }
    askBtn.disabled = true;
    resultEl.hidden = true;
    const engineEl = document.querySelector('input[name="rag-engine"]:checked');
    const engine = engineEl ? engineEl.value : "none";
    const label = engine === "gemini" ? "Retrieving passages and asking Gemini…" : engine === "local" ? "Retrieving passages and generating with Qwen locally…" : "Retrieving passages…";
    const stop = startBusy(status, label);
    const res = await postJSON("/api/exp9/ask", { question: q, engine: engine }, 15 * 60 * 1000);
    stop();
    askBtn.disabled = false;
    if (!res.ok) { setStatus(status, "error", res.error); return; }
    status.hidden = true;

    const d = res.data;
    const g = d.generation || {};
    const who = g.used ? (g.engine === "gemini" ? "Answered by Gemini (cloud): " + (g.model || "") : "Answered by local " + (g.model || "model")) : "";
    answerEl.textContent = d.answer ? d.answer : "No generated answer. See the retrieved passages below.";
    const note = g.used ? who + (g.provider ? ". " + g.provider : "") : (g.reason || "");
    noteEl.hidden = !note;
    noteEl.textContent = note;
    sourcesEl.textContent = "";
    if (!d.sources.length) {
      const li = document.createElement("li");
      li.textContent = "No relevant passages were found.";
      sourcesEl.appendChild(li);
    }
    d.sources.forEach((s) => {
      const li = document.createElement("li");
      const head = document.createElement("div");
      head.className = "src-head";
      head.textContent = s.label + " — " + s.source + " · " + s.location + " · similarity " + s.similarity;
      const text = document.createElement("p");
      text.className = "src-text";
      text.textContent = s.excerpt;
      li.appendChild(head);
      li.appendChild(text);
      sourcesEl.appendChild(li);
    });
    resultEl.hidden = false;
  });
})();
