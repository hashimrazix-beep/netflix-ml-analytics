// Animated light beams behind the app, adapted from the "BeamsBackground" React
// component. Runs inside a Streamlit component iframe and draws on a canvas
// attached to the parent page so it sits behind all app content.
//
// Performance: each beam is pre-rendered once (blur baked in) to a small
// sprite, the canvas renders at 1/4 resolution and is scaled up by CSS, and
// the loop pauses while the tab is hidden. No per-frame filters.
(function () {
  const doc = window.parent.document;
  const win = window.parent;
  if (doc.getElementById("beams-bg")) return; // already running
  if (win.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  // Holst palette: steel, sage, sand, light steel (rgb)
  const COLORS = [[65, 90, 119], [119, 141, 122], [212, 196, 168], [143, 169, 200]];
  const BEAM_COUNT = 18;
  const SCALE = 0.25; // render resolution relative to CSS pixels

  const canvas = doc.createElement("canvas");
  canvas.id = "beams-bg";
  Object.assign(canvas.style, {
    position: "fixed", inset: "0", width: "100vw", height: "100vh",
    zIndex: "-1", pointerEvents: "none",
  });
  doc.body.prepend(canvas);
  const ctx = canvas.getContext("2d", { alpha: false });

  // One blurred sprite per colour, reused by every beam of that colour.
  function makeSprite([r, g, b]) {
    const sw = 64, sh = 256, pad = 24;
    const s = doc.createElement("canvas");
    s.width = sw + pad * 2; s.height = sh + pad * 2;
    const sc = s.getContext("2d");
    const grad = sc.createLinearGradient(0, pad, 0, pad + sh);
    [[0, 0], [0.1, 0.5], [0.4, 1], [0.6, 1], [0.9, 0.5], [1, 0]].forEach(([stop, k]) =>
      grad.addColorStop(stop, `rgba(${r}, ${g}, ${b}, ${k})`));
    sc.filter = "blur(10px)";
    sc.fillStyle = grad;
    sc.fillRect(pad, pad, sw, sh);
    return s;
  }
  const sprites = COLORS.map(makeSprite);

  let w = 0, h = 0, beams = [];

  function createBeam(i, spread) {
    return {
      x: Math.random() * w * 1.5 - w * 0.25,
      y: spread ? Math.random() * h * 1.5 - h * 0.25 : h + 50,
      width: (60 + Math.random() * 80) * SCALE * 2,
      length: h * 2.5,
      angle: ((-35 + Math.random() * 10) * Math.PI) / 180,
      speed: (0.5 + Math.random() * 0.7) * SCALE * 60, // px per second at render scale
      opacity: 0.18 + Math.random() * 0.14,
      sprite: sprites[i % sprites.length],
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 1.2 + Math.random() * 1.8, // radians per second
    };
  }

  function resize() {
    w = Math.max(1, Math.round(win.innerWidth * SCALE));
    h = Math.max(1, Math.round(win.innerHeight * SCALE));
    canvas.width = w; canvas.height = h;
    beams = Array.from({ length: BEAM_COUNT }, (_, i) => createBeam(i, true));
  }

  let last = 0, running = true;
  function frame(now) {
    if (!running) return;
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;

    ctx.globalAlpha = 1;
    ctx.fillStyle = "#0D1B2A";
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = "lighter";

    beams.forEach((beam, i) => {
      beam.y -= beam.speed * dt;
      beam.pulse += beam.pulseSpeed * dt;
      if (beam.y + beam.length < -50) {
        const spacing = w / 3;
        Object.assign(beam, createBeam(i, false));
        beam.x = (i % 3) * spacing + spacing / 2 + (Math.random() - 0.5) * spacing * 0.5;
      }
      ctx.save();
      ctx.translate(beam.x, beam.y);
      ctx.rotate(beam.angle);
      ctx.globalAlpha = beam.opacity * (0.8 + Math.sin(beam.pulse) * 0.2) * 0.8;
      ctx.drawImage(beam.sprite, -beam.width / 2, 0, beam.width, beam.length);
      ctx.restore();
    });

    ctx.globalCompositeOperation = "source-over";
    win.requestAnimationFrame(frame);
  }

  doc.addEventListener("visibilitychange", () => {
    running = !doc.hidden;
    if (running) { last = 0; win.requestAnimationFrame(frame); }
  });
  let resizeTimer;
  win.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resize, 150);
  });

  resize();
  win.requestAnimationFrame(frame);
})();
