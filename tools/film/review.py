"""Contact sheet of recordings: N evenly spaced frames per recording, cursor drawn as a dot."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(sys.argv[1]); names = sys.argv[3:] or sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists())
N = 6; TW, TH = 400, 250
sheet = Image.new("RGB", (N * TW, len(names) * (TH + 18)), (20, 20, 20)); d = ImageDraw.Draw(sheet)
for row, n in enumerate(names):
    m = json.loads((root / n / "manifest.json").read_text()); fr = m["frames"]
    d.text((4, row * (TH + 18) + 2), f"{n}  {len(fr)} frames", fill=(230, 230, 230))
    for k in range(N):
        i = min(len(fr) - 1, round(k * (len(fr) - 1) / (N - 1)))
        f, cx, cy, vis, clk = fr[i]
        im = Image.open(root / n / m["files"][f]).convert("RGB").resize((TW, TH))
        if vis:
            x, y = cx / m["w"] * TW, cy / m["h"] * TH
            ImageDraw.Draw(im).ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 210, 80))
        sheet.paste(im, (k * TW, row * (TH + 18) + 16))
sheet.save(sys.argv[2]); print("saved", sys.argv[2], sheet.size)
