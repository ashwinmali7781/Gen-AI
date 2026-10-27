/* LabSpace shared UI: navigation, dashboard search, lightbox, fetch helpers, assistant. */
(function () {
  "use strict";

  // ---------- shared helpers (used by the experiment scripts) ----------
  async function request(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs || 60000);
    try {
      const res = await fetch(url, Object.assign({ signal: controller.signal }, options));
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON body */ }
      if (!res.ok || !data || data.ok === false) {
        const msg = (data && data.error) || ("Request failed (HTTP " + res.status + ").");
        return { ok: false, status: res.status, error: msg, data: data };
      }
      return { ok: true, status: res.status, data: data };
    } catch (err) {
      if (err && err.name === "AbortError") {
        return { ok: false, status: 0, error: "The request timed out. The server may still be working; try again shortly." };
      }
      return { ok: false, status: 0, error: "Could not reach the server. Is it still running?" };
    } finally {
      clearTimeout(timer);
    }
  }

  function postJSON(url, body, timeoutMs) {
    return request(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }, timeoutMs);
  }

  // Busy indicator with a real elapsed-time counter (no fake progress bar).
  function setStatus(el, kind, text) {
    if (!el) return;
    el.hidden = false;
    el.className = "status-line" + (kind === "error" ? " error" : kind === "success" ? " success" : "");
    el.textContent = "";
    if (kind === "busy") {
      const spin = document.createElement("span");
      spin.className = "spinner";
      spin.setAttribute("aria-hidden", "true");
      el.appendChild(spin);
    }
    const span = document.createElement("span");
    span.textContent = text;
    el.appendChild(span);
  }

  function startBusy(el, label) {
    const t0 = Date.now();
    setStatus(el, "busy", label + " 0s");
    const id = setInterval(() => setStatus(el, "busy", label + " " + Math.round((Date.now() - t0) / 1000) + "s"), 1000);
    return function stop() { clearInterval(id); };
  }

  window.LabSpace = { request, postJSON, setStatus, startBusy };

  // ---------- sidebar (mobile) ----------
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("menu-toggle");
  const backdrop = document.getElementById("sidebar-backdrop");
  function setSidebar(open) {
    if (!sidebar || !toggle) return;
    sidebar.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
    if (backdrop) backdrop.hidden = !open;
  }
  if (toggle) toggle.addEventListener("click", () => setSidebar(!sidebar.classList.contains("open")));
  if (backdrop) backdrop.addEventListener("click", () => setSidebar(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { setSidebar(false); closeLightbox(); } });

  // ---------- dashboard search + filter ----------
  const grid = document.getElementById("exp-grid");
  if (grid) {
    const input = document.getElementById("exp-search");
    const empty = document.getElementById("exp-empty");
    const count = document.getElementById("exp-count");
    const buttons = Array.from(document.querySelectorAll(".seg-btn"));
    const cards = Array.from(grid.querySelectorAll(".card"));
    let filter = "all";

    function apply() {
      const q = (input.value || "").trim().toLowerCase();
      let shown = 0;
      cards.forEach((card) => {
        const kind = card.dataset.kind;
        const kindOk = filter === "all" ||
          (filter === "live" && (kind === "live" || kind === "hybrid")) ||
          (filter === "saved" && (kind === "saved" || kind === "hybrid"));
        const textOk = !q || card.dataset.search.indexOf(q) !== -1;
        const visible = kindOk && textOk;
        card.hidden = !visible;
        if (visible) shown++;
      });
      empty.hidden = shown !== 0;
      count.textContent = "Showing " + shown + " of " + cards.length + " experiments.";
    }
    input.addEventListener("input", apply);
    buttons.forEach((b) => b.addEventListener("click", () => {
      filter = b.dataset.filter;
      buttons.forEach((x) => { const on = x === b; x.classList.toggle("active", on); x.setAttribute("aria-pressed", String(on)); });
      apply();
    }));
    apply();
  }

  // ---------- lightbox ----------
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  const lbCap = document.getElementById("lightbox-caption");
  const lbClose = document.getElementById("lightbox-close");
  let lastFocus = null;

  function openLightbox(src, caption) {
    if (!lb) return;
    lastFocus = document.activeElement;
    lbImg.src = src;
    lbImg.alt = caption || "";
    lbCap.textContent = caption || "";
    lb.hidden = false;
    lbClose.focus();
  }
  function closeLightbox() {
    if (!lb || lb.hidden) return;
    lb.hidden = true;
    lbImg.removeAttribute("src");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-lightbox]");
    if (t) {
      const img = t.querySelector("img");
      openLightbox(t.dataset.full || (img && img.src), t.dataset.caption || (img && img.alt));
      return;
    }
    if (lb && !lb.hidden && (e.target === lb || e.target === lbClose)) closeLightbox();
  });

  // ---------- optional Gemini assistant ----------
  const aForm = document.getElementById("assistant-form");
  if (aForm) {
    const q = document.getElementById("assistant-q");
    const btn = document.getElementById("assistant-btn");
    const out = document.getElementById("assistant-out");
    aForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const question = q.value.trim();
      if (!question) { out.hidden = false; out.textContent = "Please enter a question."; return; }
      btn.disabled = true;
      out.hidden = false;
      out.textContent = "Thinking…";
      const res = await postJSON("/api/assistant/ask", { question }, 60000);
      btn.disabled = false;
      out.textContent = res.ok ? res.data.answer + "\n\n— " + res.data.provider : "Error: " + res.error;
    });
  }
})();
