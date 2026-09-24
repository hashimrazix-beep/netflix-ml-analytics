// Page interactions for the Streamlit app, injected into the parent page:
// 3D card tilt, scroll-reveal, micro-interactions, and parallax.
// Transform/opacity only, batched in requestAnimationFrame.
(function () {
  const doc = window.parent.document;
  const win = window.parent;
  if (win.__movieInteractions) return; // already running
  win.__movieInteractions = true;
  const reduced = win.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = win.matchMedia("(pointer: fine)").matches;

  // 1. The intro loader is rendered by app.py with the theme CSS so it shows first.

  if (reduced) return;

  // ---------- 3. Entrance reveal: fade/slide elements in as they scroll into view ----------
  const REVEAL = ".card, .section, .label, .strip, div[data-testid='stPlotlyChart'], div[data-testid='stDataFrame']";
  const io = new win.IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("revealed"); io.unobserve(e.target); }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
  function scan() {
    doc.querySelectorAll(REVEAL).forEach((el) => {
      if (el.dataset.reveal) return;
      el.dataset.reveal = "1";
      el.classList.add("reveal");
      io.observe(el);
    });
  }
  new win.MutationObserver(scan).observe(doc.body, { childList: true, subtree: true });
  scan();

  // ---------- 2. 3D motion: cards tilt toward the pointer with a moving glare ----------
  let tiltTarget = null, tiltEvent = null, tiltQueued = false;
  function applyTilt() {
    tiltQueued = false;
    if (!tiltTarget || !tiltEvent) return;
    const r = tiltTarget.getBoundingClientRect();
    const px = (tiltEvent.clientX - r.left) / r.width;
    const py = (tiltEvent.clientY - r.top) / r.height;
    tiltTarget.style.setProperty("--ry", `${(px - 0.5) * 10}deg`);
    tiltTarget.style.setProperty("--rx", `${(0.5 - py) * 8}deg`);
    tiltTarget.style.setProperty("--mx", `${px * 100}%`);
    tiltTarget.style.setProperty("--my", `${py * 100}%`);
  }
  function resetTilt(el) {
    el.style.setProperty("--rx", "0deg");
    el.style.setProperty("--ry", "0deg");
  }
  if (finePointer) {
    doc.addEventListener("pointermove", (e) => {
      const card = e.target.closest && e.target.closest(".card");
      if (card !== tiltTarget) {
        if (tiltTarget) resetTilt(tiltTarget);
        tiltTarget = card;
      }
      tiltEvent = e;
      if (card && !tiltQueued) { tiltQueued = true; win.requestAnimationFrame(applyTilt); }
    }, { passive: true });
    doc.addEventListener("pointerleave", () => { if (tiltTarget) resetTilt(tiltTarget); tiltTarget = null; });
  }

  // ---------- 4. Micro-interactions: ripple on buttons, cards, and nav pills ----------
  doc.addEventListener("pointerdown", (e) => {
    const el = e.target.closest && e.target.closest(
      "button, details.card summary, div[data-testid='stButtonGroup'] button");
    if (!el) return;
    const host = el.tagName === "SUMMARY" ? el.parentElement : el;
    const r = host.getBoundingClientRect();
    const size = Math.max(r.width, r.height) * 1.2;
    const ripple = doc.createElement("span");
    ripple.className = "ripple";
    Object.assign(ripple.style, {
      width: `${size}px`, height: `${size}px`,
      left: `${e.clientX - r.left - size / 2}px`, top: `${e.clientY - r.top - size / 2}px`,
    });
    host.classList.add("has-ripple");
    host.appendChild(ripple);
    setTimeout(() => ripple.remove(), 650);
  }, { passive: true });

  // ---------- 5. Parallax: hero drifts on scroll, background follows the pointer ----------
  const scroller = doc.querySelector("section[data-testid='stMain']") || doc.scrollingElement;
  let px = 0, py = 0, sy = 0, parallaxQueued = false;
  function applyParallax() {
    parallaxQueued = false;
    const hero = doc.querySelector(".hero");
    if (hero) {
      hero.style.transform = `translate3d(0, ${sy * 0.35}px, 0)`;
      hero.style.opacity = String(Math.max(0, 1 - sy / 420));
    }
    const bg = doc.getElementById("beams-bg");
    if (bg) bg.style.transform = `translate3d(${px * -18}px, ${py * -18 - sy * 0.08}px, 0) scale(1.06)`;
  }
  const queue = () => { if (!parallaxQueued) { parallaxQueued = true; win.requestAnimationFrame(applyParallax); } };
  doc.addEventListener("scroll", (e) => {
    const t = e.target === doc ? doc.scrollingElement : e.target;
    sy = t && t.scrollTop ? t.scrollTop : (scroller ? scroller.scrollTop : 0);
    queue();
  }, { passive: true, capture: true });
  if (finePointer) {
    doc.addEventListener("pointermove", (e) => {
      px = e.clientX / win.innerWidth - 0.5;
      py = e.clientY / win.innerHeight - 0.5;
      queue();
    }, { passive: true });
  }
})();
