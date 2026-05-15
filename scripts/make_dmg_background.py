#!/usr/bin/env python3
"""
Genera assets/dmg_background.png — sfondo scuro 560×400 per la finestra DMG.
Usato da create-dmg nel workflow GitHub Actions.
"""

import os

os.makedirs("assets", exist_ok=True)

W, H = 560, 400

SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">
  <rect width="{W}" height="{H}" fill="#0f0f1a"/>

  <!-- Subtle grid lines -->
  {"".join(
      f'<line x1="{x}" y1="0" x2="{x}" y2="{H}" stroke="#ffffff" stroke-width="0.3" opacity="0.04"/>'
      for x in range(0, W, 40)
  )}
  {"".join(
      f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="#ffffff" stroke-width="0.3" opacity="0.04"/>'
      for y in range(0, H, 40)
  )}

  <!-- Waveform decoration bottom -->
  <polyline points="0,340 40,320 80,350 120,310 160,355 200,325 240,345 280,315 320,350 360,320 400,345 440,318 480,348 520,322 560,340"
    fill="none" stroke="#5a3aaf" stroke-width="1.5" opacity="0.35"/>
  <polyline points="0,360 40,345 80,365 120,338 160,368 200,348 240,362 280,340 320,365 360,342 400,360 440,338 480,362 520,344 560,358"
    fill="none" stroke="#3a1e8c" stroke-width="1" opacity="0.25"/>

  <!-- App name bottom center -->
  <text x="{W//2}" y="{H-18}" text-anchor="middle"
    font-family="-apple-system, Helvetica Neue, sans-serif"
    font-size="12" fill="#ffffff" opacity="0.25">StemForge 1.0.0</text>
</svg>"""

svg_path = "assets/dmg_background.svg"
png_path = "assets/dmg_background.png"

with open(svg_path, "w") as f:
    f.write(SVG)

# Try cairosvg
try:
    import cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=W, output_height=H)
    print(f"✓ {png_path} (cairosvg)")
except ImportError:
    ret = os.system(f"rsvg-convert -w {W} -h {H} '{svg_path}' -o '{png_path}'")
    if ret == 0:
        print(f"✓ {png_path} (rsvg-convert)")
    else:
        # Pillow fallback — plain dark background
        try:
            from PIL import Image
            img = Image.new("RGB", (W, H), (15, 15, 26))
            img.save(png_path)
            print(f"✓ {png_path} (Pillow fallback, no decoration)")
        except Exception as e:
            print(f"Warning: {e} — create-dmg will use no background")

os.remove(svg_path)
