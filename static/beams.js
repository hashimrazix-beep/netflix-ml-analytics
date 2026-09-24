// Animated light beams behind the app, adapted from the "BeamsBackground" React
// component. It runs inside a Streamlit component iframe and draws on a canvas
// attached to the parent page so it sits behind all app content.
(function () {
  const doc = window.parent.document;
  const win = window.parent;
  if (doc.getElementById("beams-bg")) return; // already running
  if (win.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  // Holst palette: steel, sage, sand, light steel (rgb)
  const COLORS = [[65, 90, 119], [119, 141, 122], [212, 196, 168], [143, 169, 200]];
  const BEAM_COUNT = 30;

  const canvas = doc.createElement("canvas");
  canvas.id = "beams-bg";
  Object.assign(canvas.style, {
    position: "fixed", inset: "0", zIndex: "-1", pointerEvents: "none",
    filter: "blur(15px)", opacity: "0.9",
  });
  doc.body.prepend(canvas);
  const ctx = canvas.getContext("2d");
  let w = 0, h = 0;

  function createBeam(i) {
    return {
      x: Math.random() * w * 1.5 - w * 0.25,
      y: Math.random() * h * 1.5 - h * 0.25,
      width: 30 + Math.random() * 60,
      length: h * 2.5,
      angle: -35 + Math.random() * 10,
      speed: 0.6 + Math.random() * 1.2,
      opacity: 0.12 + Math.random() * 0.16,
      color: COLORS[i % COLORS.length],
      pulse: Math.random() * Math.PI * 2,
      pulseSpeed: 0.02 + Math.random() * 0.03,
    };
  }

  let beams = [];
  function resize() {
    const dpr = win.devicePixelRatio || 1;
    w = win.innerWidth; h = win.innerHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    beams = Array.from({ length: BEAM_COUNT }, (_, i) => createBeam(i));
  }

  function resetBeam(beam, i) {
    const spacing = w / 3;
    beam.y = h + 100;
    beam.x = (i % 3) * spacing + spacing / 2 + (Math.random() - 0.5) * spacing * 0.5;
    beam.width = 100 + Math.random() * 100;
    beam.speed = 0.5 + Math.random() * 0.4;
    beam.opacity = 0.2 + Math.random() * 0.1;
  }

  function drawBeam(beam) {
    ctx.save();
    ctx.translate(beam.x, beam.y);
    ctx.rotate((beam.angle * Math.PI) / 180);
    const a = beam.opacity * (0.8 + Math.sin(beam.pulse) * 0.2) * 0.85;
    const [r, g, b] = beam.color;
    const grad = ctx.createLinearGradient(0, 0, 0, beam.length);
    [[0, 0], [0.1, 0.5], [0.4, 1], [0.6, 1], [0.9, 0.5], [1, 0]].forEach(([stop, k]) =>
      grad.addColorStop(stop, `rgba(${r}, ${g}, ${b}, ${a * k})`));
    ctx.fillStyle = grad;
    ctx.fillRect(-beam.width / 2, 0, beam.width, beam.length);
    ctx.restore();
  }

  function animate() {
    ctx.clearRect(0, 0, w, h);
    ctx.filter = "blur(35px)";
    beams.forEach((beam, i) => {
      beam.y -= beam.speed;
      beam.pulse += beam.pulseSpeed;
      if (beam.y + beam.length < -100) resetBeam(beam, i);
      drawBeam(beam);
    });
    win.requestAnimationFrame(animate);
  }

  resize();
  win.addEventListener("resize", resize);
  animate();
})();
