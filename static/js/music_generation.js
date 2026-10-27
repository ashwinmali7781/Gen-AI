/* Experiment 8 — local MusicGen generation */
(function () {
  "use strict";
  const { postJSON, setStatus, startBusy } = window.LabSpace;
  const form = document.getElementById("music-form");
  if (!form) return;
  const prompt = document.getElementById("music-prompt");
  const seconds = document.getElementById("music-seconds");
  const btn = document.getElementById("music-btn");
  const status = document.getElementById("music-status");
  const result = document.getElementById("music-result");
  const audio = document.getElementById("music-audio");
  const caption = document.getElementById("music-caption");
  const download = document.getElementById("music-download");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = prompt.value.trim();
    if (!text) {
      setStatus(status, "error", "Please enter a music description.");
      prompt.focus();
      return;
    }
    btn.disabled = true;
    result.hidden = true;
    const stop = startBusy(status, "Generating music (first run downloads the model; CPU is slow)…");
    const res = await postJSON("/api/exp8/generate", { prompt: text, seconds: Number(seconds.value) }, 20 * 60 * 1000);
    stop();
    btn.disabled = false;

    if (!res.ok) {
      setStatus(status, "error", res.error);
      return;
    }
    audio.src = res.data.url;
    caption.textContent = "“" + res.data.prompt + "” — " + res.data.seconds_audio + "s of audio in " + res.data.seconds_taken + "s on " + res.data.device;
    download.href = res.data.download_url;
    result.hidden = false;
    setStatus(status, "success", "Music generated.");
  });
})();
