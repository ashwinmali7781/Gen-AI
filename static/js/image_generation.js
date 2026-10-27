/* Experiment 4 — hosted FLUX.1-schnell image generation */
(function () {
  "use strict";
  const { postJSON, setStatus, startBusy } = window.LabSpace;
  const form = document.getElementById("img-form");
  if (!form) return;
  const prompt = document.getElementById("img-prompt");
  const size = document.getElementById("img-size");
  const btn = document.getElementById("img-btn");
  const status = document.getElementById("img-status");
  const result = document.getElementById("img-result");
  const out = document.getElementById("img-out");
  const thumb = document.getElementById("img-thumb");
  const caption = document.getElementById("img-caption");
  const download = document.getElementById("img-download");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = prompt.value.trim();
    if (!text) {
      setStatus(status, "error", "Please enter a prompt.");
      prompt.focus();
      return;
    }
    btn.disabled = true;
    result.hidden = true;
    const stop = startBusy(status, "Generating image on the hosted provider…");
    const res = await postJSON("/api/exp4/generate", { prompt: text, size: Number(size.value) }, 3 * 60 * 1000);
    stop();
    btn.disabled = false;

    if (!res.ok) {
      setStatus(status, "error", res.error);
      return;
    }
    out.src = res.data.url;
    thumb.dataset.full = res.data.url;
    thumb.dataset.caption = res.data.prompt;
    caption.textContent = res.data.model + " · " + res.data.size + "×" + res.data.size + " · “" + res.data.prompt + "”";
    download.href = res.data.download_url;
    result.hidden = false;
    setStatus(status, "success", "Image generated.");
  });
})();
