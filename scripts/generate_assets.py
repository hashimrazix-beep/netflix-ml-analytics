"""
Generate static site assets: favicon set, Open Graph image, sitemap.xml, robots.txt.

Usage:
    python scripts/generate_assets.py [--url https://your-app.streamlit.app]
Images are written compressed (palette PNGs for icons, progressive JPEG for OG).
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.site import DEFAULT_APP_URL, PAGES, page_url  # noqa: E402

STATIC = PROJECT_ROOT / "static"
NAVY, INK, STEEL, SAGE, SAND, CREAM = "#0D1B2A", "#1B263B", "#415A77", "#778D7A", "#D4C4A8", "#F4F1DE"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def beams(size, alpha=150):
    """Soft diagonal light beams in the palette, like the app background."""
    w, h = size
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, (x, color) in enumerate([(0.1, STEEL), (0.35, SAGE), (0.62, STEEL), (0.85, SAND)]):
        rgb = Image.new("RGB", (1, 1), color).getpixel((0, 0))
        x0 = int(x * w)
        bw = int(w * 0.07)
        d.polygon([(x0, h), (x0 + bw, h), (x0 + bw + int(h * 0.7), 0), (x0 + int(h * 0.7), 0)],
                  fill=(*rgb, alpha if i != 3 else alpha // 2))
    return layer.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 25))


def icon(size: int) -> Image.Image:
    """Rounded navy tile with a sand 'M' and a sage underline."""
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=NAVY)
    f = font(BOLD, int(s * 0.58))
    bbox = d.textbbox((0, 0), "M", font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((s - tw) / 2 - bbox[0], (s - th) / 2 - bbox[1] - s * 0.05), "M", font=f, fill=SAND)
    d.rounded_rectangle([s * 0.28, s * 0.76, s * 0.72, s * 0.81], radius=int(s * 0.03), fill=SAGE)
    return img.resize((size, size), Image.LANCZOS)


def save_png(img: Image.Image, path: Path) -> None:
    img.quantize(colors=64, method=Image.Quantize.FASTOCTREE).save(path, optimize=True)


def og_image(path: Path) -> None:
    w, h = 1200, 630
    img = Image.new("RGB", (w, h), NAVY)
    img.paste(beams((w, h)), (0, 0), beams((w, h)))
    d = ImageDraw.Draw(img)
    img.paste(icon(96), (96, 150), icon(96))
    d.text((96, 280), "Movie ML Analytics", font=font(BOLD, 78), fill=CREAM)
    d.text((98, 385), "Recommendations, predictions, and a map of", font=font(REGULAR, 34), fill=SAND)
    d.text((98, 430), "8,790 films and series, powered by machine learning.", font=font(REGULAR, 34), fill=SAND)
    d.rounded_rectangle([96, 510, 360, 516], radius=3, fill=SAGE)
    img.save(path, "JPEG", quality=82, optimize=True, progressive=True)


def sitemap(app_url: str) -> str:
    urls = "\n".join(
        f"  <url><loc>{page_url(app_url, p).replace('&', '&amp;')}</loc>"
        f"<changefreq>monthly</changefreq><priority>{'1.0' if p.slug == 'discover' else '0.6'}</priority></url>"
        for p in PAGES.values() if p.indexable
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls}\n</urlset>\n'


def robots(app_url: str) -> str:
    return f"User-agent: *\nAllow: /\nDisallow: /?page=thanks\n\nSitemap: {app_url.rstrip('/')}/app/static/sitemap.xml\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_APP_URL)
    args = parser.parse_args()
    STATIC.mkdir(exist_ok=True)
    save_png(icon(32), STATIC / "favicon-32.png")
    save_png(icon(192), STATIC / "favicon-192.png")
    save_png(icon(180), STATIC / "apple-touch-icon.png")
    save_png(icon(512), STATIC / "icon-512.png")
    og_image(STATIC / "og-image.jpg")
    (STATIC / "sitemap.xml").write_text(sitemap(args.url))
    (STATIC / "robots.txt").write_text(robots(args.url))
    for f in sorted(STATIC.glob("*")):
        if f.suffix in {".png", ".jpg", ".xml", ".txt"}:
            print(f"{f.name:22s} {f.stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
