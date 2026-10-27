/* Experiments 6 & 7 — replay of SAVED epoch snapshots (no training or generation happens). */
(function () {
  "use strict";
  const root = document.getElementById("replay");
  if (!root) return;
  let frames = [];
  try { frames = JSON.parse(root.dataset.frames || "[]"); } catch (_) { frames = []; }
  if (!frames.length) return;

  const img = document.getElementById("replay-img");
  const cap = document.getElementById("replay-caption");
  const play = document.getElementById("replay-play");
  const reset = document.getElementById("replay-reset");
  const speed = document.getElementById("replay-speed");
  const slider = document.getElementById("replay-slider");
  let i = 0, timer = null;

  // Preload so playback does not flicker.
  frames.forEach((f) => { const im = new Image(); im.src = f.url; });

  function show(n) {
    i = Math.max(0, Math.min(frames.length - 1, n));
    img.src = frames[i].url;
    img.alt = frames[i].label + " snapshot";
    cap.textContent = frames[i].label + " (saved snapshot)";
    slider.value = String(i);
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    play.textContent = i >= frames.length - 1 ? "Play again" : "Play replay";
  }
  function start() {
    if (i >= frames.length - 1) show(0);
    play.textContent = "Pause";
    timer = setInterval(() => {
      if (i >= frames.length - 1) { stop(); return; }
      show(i + 1);
      if (i >= frames.length - 1) stop();
    }, Number(speed.value));
  }
  play.addEventListener("click", () => (timer ? stop() : start()));
  reset.addEventListener("click", () => { stop(); show(0); play.textContent = "Play replay"; });
  speed.addEventListener("change", () => { if (timer) { stop(); start(); } });
  slider.addEventListener("input", () => { stop(); show(Number(slider.value)); });
  show(0);
})();
