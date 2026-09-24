// Site-level head management for the Streamlit app, run in the parent page:
// meta description / robots / Open Graph / Twitter tags, canonical link,
// favicon links, JSON-LD, a cookie consent banner, and consent-gated GA4.
// Expects window.__PAGE__ (set by app.py) in this iframe.
(function () {
  const cfg = window.__PAGE__;
  const doc = window.parent.document;
  const win = window.parent;
  if (!cfg) return;

  // ---------- Head tags ----------
  function upsert(selector, create, attrs) {
    let el = doc.head.querySelector(selector);
    if (!el) { el = doc.createElement(create); doc.head.appendChild(el); }
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
  }
  const meta = (key, value, attr = "name") =>
    upsert(`meta[${attr}="${key}"]`, "meta", { [attr]: key, content: value });

  doc.title = cfg.title;
  meta("description", cfg.description);
  meta("robots", cfg.indexable ? "index, follow" : "noindex, follow");
  meta("theme-color", "#0D1B2A");
  meta("og:type", "website", "property");
  meta("og:site_name", cfg.siteName, "property");
  meta("og:title", cfg.title, "property");
  meta("og:description", cfg.description, "property");
  meta("og:url", cfg.url, "property");
  meta("og:image", cfg.ogImage, "property");
  meta("og:image:width", "1200", "property");
  meta("og:image:height", "630", "property");
  meta("og:image:alt", cfg.ogImageAlt, "property");
  meta("twitter:card", "summary_large_image");
  meta("twitter:title", cfg.title);
  meta("twitter:description", cfg.description);
  meta("twitter:image", cfg.ogImage);
  meta("twitter:image:alt", cfg.ogImageAlt);
  upsert('link[rel="canonical"]', "link", { rel: "canonical", href: cfg.url });
  upsert('link[rel="apple-touch-icon"]', "link", { rel: "apple-touch-icon", sizes: "180x180", href: cfg.static + "apple-touch-icon.png" });
  upsert('link[rel="icon"][sizes="192x192"]', "link", { rel: "icon", type: "image/png", sizes: "192x192", href: cfg.static + "favicon-192.png" });
  upsert('link[rel="icon"][sizes="512x512"]', "link", { rel: "icon", type: "image/png", sizes: "512x512", href: cfg.static + "icon-512.png" });

  let ld = doc.getElementById("ld-json");
  if (!ld) { ld = doc.createElement("script"); ld.type = "application/ld+json"; ld.id = "ld-json"; doc.head.appendChild(ld); }
  ld.textContent = JSON.stringify(cfg.jsonLd);

  // ---------- Consent + GA4 ----------
  const KEY = "mma-cookie-consent";
  const read = () => { try { return win.localStorage.getItem(KEY); } catch (e) { return null; } };
  const write = (v) => { try { win.localStorage.setItem(KEY, v); } catch (e) { /* storage blocked */ } };

  function loadAnalytics() {
    if (!cfg.gaId) return;
    if (!win.__gaLoaded) {
      win.__gaLoaded = true;
      win.dataLayer = win.dataLayer || [];
      win.gtag = function () { win.dataLayer.push(arguments); };
      win.gtag("js", new Date());
      win.gtag("config", cfg.gaId, { send_page_view: false, anonymize_ip: true });
      const s = doc.createElement("script");
      s.async = true;
      s.src = "https://www.googletagmanager.com/gtag/js?id=" + encodeURIComponent(cfg.gaId);
      doc.head.appendChild(s);
    }
    if (win.__gaLastPage !== cfg.slug) {
      win.__gaLastPage = cfg.slug;
      win.gtag("event", "page_view", { page_title: cfg.title, page_location: cfg.url, page_path: "/" + (cfg.slug === "discover" ? "" : "?page=" + cfg.slug) });
    }
  }

  function showBanner() {
    if (doc.getElementById("cookie-banner")) return;
    const bar = doc.createElement("div");
    bar.id = "cookie-banner";
    bar.setAttribute("role", "dialog");
    bar.setAttribute("aria-label", "Cookie consent");
    bar.innerHTML =
      '<p>We use optional analytics cookies to see which pages are useful. ' +
      '<a href="?page=privacy" target="_self">Privacy policy</a></p>' +
      '<div class="cookie-actions"><button type="button" data-choice="declined">Decline</button>' +
      '<button type="button" data-choice="accepted" class="primary">Accept</button></div>';
    bar.addEventListener("click", (e) => {
      const choice = e.target.getAttribute && e.target.getAttribute("data-choice");
      if (!choice) return;
      write(choice);
      bar.classList.add("hide");
      setTimeout(() => bar.remove(), 400);
      if (choice === "accepted") loadAnalytics();
    });
    doc.body.appendChild(bar);
  }

  // Footer "Cookie settings" link reopens the banner
  if (!win.__cookieLinkBound) {
    win.__cookieLinkBound = true;
    doc.addEventListener("click", (e) => {
      const link = e.target.closest && e.target.closest("[data-cookie-settings]");
      if (!link) return;
      e.preventDefault();
      try { win.localStorage.removeItem(KEY); } catch (err) { /* ignore */ }
      showBanner();
    });
  }

  const consent = read();
  if (consent === "accepted") loadAnalytics();
  else if (consent !== "declined") showBanner();
})();
