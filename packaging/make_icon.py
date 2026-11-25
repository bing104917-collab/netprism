"""Generate the NetPrism application icon (ICO + PNG).

Usage::

    python packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
OUT_DIR = Path(__file__).resolve().parent
ROOT = OUT_DIR.parent


def rounded_background() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = 28
    draw.rounded_rectangle(
        [margin, margin, SIZE - margin, SIZE - margin],
        radius=210,
        fill=(6, 11, 20, 255),
        outline=(28, 58, 86, 255),
        width=8,
    )
    # Soft vignette / glow in the middle.
    glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([170, 150, 854, 834], fill=(20, 120, 180, 40))
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    image = Image.alpha_composite(image, glow)
    return image


def draw_prism(image: Image.Image) -> None:
    overlay = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    top = (512, 170)
    left = (176, 824)
    right = (848, 824)
    middle = (512, 588)

    # Back face glow.
    draw.polygon([top, right, left], fill=(56, 189, 248, 22))
    # Left face: cyan, right face: violet.
    draw.polygon([top, middle, left], fill=(34, 211, 238, 76))
    draw.polygon([top, right, middle], fill=(167, 139, 250, 82))
    # Edges.
    for a, b in ((top, left), (left, right), (right, top), (top, middle), (middle, left), (middle, right)):
        draw.line([a, b], fill=(186, 230, 253, 210), width=10, joint="curve")

    # Refracted beam lines.
    draw.line([middle, (512, 890)], fill=(125, 211, 252, 160), width=6)
    draw.line([(430, 690), (594, 690)], fill=(192, 132, 252, 90), width=4)

    glow = overlay.filter(ImageFilter.GaussianBlur(18))
    image = Image.alpha_composite(image, glow)
    image = Image.alpha_composite(image, overlay)
    return image


def main() -> None:
    icon = rounded_background()
    icon = draw_prism(icon)
    # Add a subtle white top highlight.
    draw = ImageDraw.Draw(icon)
    draw.line([(512, 176), (188, 812)], fill=(224, 242, 254, 90), width=4)

    png_path = ROOT / "docs" / "screenshots" / "icon.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    icon.resize((512, 512), Image.LANCZOS).save(png_path)

    ico = OUT_DIR / "netprism.ico"
    icon.save(ico, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"wrote {ico} ({ico.stat().st_size} bytes)")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
