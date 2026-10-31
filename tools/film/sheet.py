"""Tile still images into one contact sheet: python3 tools/film/sheet.py OUT.png COLS WIDTH img1 img2 ..."""
import sys
from PIL import Image, ImageDraw
out, cols, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ims = sys.argv[4:]; h = w * 9 // 16; rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * (w + 6), rows * (h + 24)), (40, 40, 40)); d = ImageDraw.Draw(sheet)
for i, p in enumerate(ims):
    im = Image.open(p).convert("RGB").resize((w, h), Image.LANCZOS); x, y = (i % cols) * (w + 6), (i // cols) * (h + 24)
    sheet.paste(im, (x, y + 20)); d.text((x + 4, y + 4), p.rsplit("/", 1)[-1], fill=(230, 230, 230))
sheet.save(out); print(out, sheet.size)
