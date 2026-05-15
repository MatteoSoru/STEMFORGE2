#!/usr/bin/env python3
"""
Genera le icone StemForge in tutti i formati richiesti da iconutil.
Crea icon.iconset/ con i PNG necessari, poi build.yml esegue:
    iconutil -c icns icon.iconset -o AppIcon.icns
"""

import os
import math

SIZES = [16, 32, 64, 128, 256, 512, 1024]
ICONSET = "icon.iconset"
os.makedirs(ICONSET, exist_ok=True)


def make_icon_svg(size):
    """SVG dell'icona StemForge: sfondo viola scuro + waveform."""
    bars = [0.3, 0.6, 1.0, 0.75, 0.45, 0.8, 0.5]
    bar_w = size * 0.07
    gap = size * 0.025
    total_w = len(bars) * bar_w + (len(bars) - 1) * gap
    x0 = (size - total_w) / 2
    max_h = size * 0.55
    cy = size / 2

    rects = ""
    for i, h_ratio in enumerate(bars):
        bh = max_h * h_ratio
        bx = x0 + i * (bar_w + gap)
        by = cy - bh / 2
        rects += (
            f'<rect x="{bx:.1f}" y="{by:.1f}" '
            f'width="{bar_w:.1f}" height="{bh:.1f}" '
            f'rx="{bar_w/2:.1f}" fill="#c4a8ff"/>'
        )

    r = size * 0.22  # corner radius for app icon shape
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">
  <rect width="{size}" height="{size}" rx="{r}" fill="#1a0e3d"/>
  <rect width="{size}" height="{size}" rx="{r}" fill="#2d1a6e" opacity="0.6"/>
  {rects}
</svg>"""


def svg_to_png(svg_str, out_path, size):
    """Converti SVG → PNG usando cairosvg se disponibile, altrimenti rsvg-convert."""
    svg_path = out_path.replace(".png", ".svg")
    with open(svg_path, "w") as f:
        f.write(svg_str)
    # Prova cairosvg prima
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=out_path, output_width=size, output_height=size)
        os.remove(svg_path)
        return
    except ImportError:
        pass
    # Fallback: rsvg-convert (disponibile su macOS con brew install librsvg)
    ret = os.system(f"rsvg-convert -w {size} -h {size} '{svg_path}' -o '{out_path}'")
    os.remove(svg_path)
    if ret != 0:
        # Ultimo fallback: Pillow + aggdraw (solo forma base)
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            r = int(size * 0.22)
            draw.rounded_rectangle([(0, 0), (size, size)], radius=r, fill="#1a0e3d")
            img.save(out_path)
        except Exception as e:
            print(f"Warning: could not render icon at {size}px: {e}")


for s in SIZES:
    svg = make_icon_svg(s)
    for suffix, actual in [("", s), ("@2x", s)]:
        fname = f"{ICONSET}/icon_{s}x{s}{suffix}.png"
        svg_to_png(svg, fname, actual)
        print(f"  ✓ {fname}")

# iconutil naming: icon_16x16.png, icon_16x16@2x.png (= 32px render), etc.
# The @2x files should be the double-resolution render — fix them:
import shutil
for s in SIZES[:-1]:
    src = f"{ICONSET}/icon_{s*2}x{s*2}.png"
    dst = f"{ICONSET}/icon_{s}x{s}@2x.png"
    if os.path.exists(src):
        shutil.copy(src, dst)

print("Icon set ready →", ICONSET)
