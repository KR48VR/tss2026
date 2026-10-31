"""The film's task field: one dot per O*NET task (18,796), grouped by occupation family (SOC major group) in a
squarified treemap, each dot carrying the year AI could be doing it in Singapore (central case, from the timing
model's forecast_tasks) and the share of the task AI does once it is used (ai_share). Writes tools/film/data/field.json."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
FAMILIES = {"11": "Management", "13": "Business and finance", "15": "Computers and maths", "17": "Architecture and engineering",
            "19": "Science", "21": "Community and social service", "23": "Legal", "25": "Education", "27": "Arts, design and media",
            "29": "Health practitioners", "31": "Health support", "33": "Protective services", "35": "Food preparation and serving",
            "37": "Cleaning and grounds", "39": "Personal care", "41": "Sales", "43": "Office and admin support",
            "45": "Farming, fishing, forestry", "47": "Construction", "49": "Installation and repair", "51": "Production",
            "53": "Transport and moving", "55": "Military"}
FEATURE = "13-2011.00"   # Accountants and Auditors: the job the films follow


def squarify(sizes, x, y, w, h):
    """Bruls et al. squarified treemap; sizes sorted descending, sum normalised to the rectangle's area."""
    out = []; sizes = list(sizes); total = sum(s for _, s in sizes); scale = w * h / total
    items = [(k, s * scale) for k, s in sizes]
    def worst(row, side):
        s = sum(a for _, a in row); return max(max(side * side * a / (s * s), (s * s) / (side * side * a)) for _, a in row)
    while items:
        side = min(w, h); row = [items.pop(0)]
        while items and worst(row + [items[0]], side) <= worst(row, side):
            row.append(items.pop(0))
        s = sum(a for _, a in row)
        if w >= h:   # lay the row down the left edge
            cw = s / h; cy = y
            for k, a in row:
                ch = a / cw; out.append((k, x, cy, cw, ch)); cy += ch
            x += cw; w -= cw
        else:
            rh = s / w; cx = x
            for k, a in row:
                cw = a / rh; out.append((k, cx, y, cw, rh)); cx += cw
            y += rh; h -= rh
    return out


def main():
    ft = pd.read_parquet(P / "forecast_tasks.parquet", columns=["task_id", "onet_soc_code", "disrupt_year_p50", "ai_share", "physical", "status"])
    ft["fam"] = ft.onet_soc_code.str[:2]
    ft = ft.sort_values(["fam", "onet_soc_code", "disrupt_year_p50"]).reset_index(drop=True)
    sizes = ft.groupby("fam").size().sort_values(ascending=False)
    X0, Y0, WW, HH, GAP = 120, 150, 1680, 800, 6
    cells = squarify(list(sizes.items()), X0, Y0, WW, HH)
    xs = np.zeros(len(ft)); ys = np.zeros(len(ft)); groups = []
    for gi, (fam, cx, cy, cw, ch) in enumerate(cells):
        idx = ft.index[ft.fam == fam].to_numpy(); n = len(idx)
        ix, iy, iw, ih = cx + GAP / 2, cy + GAP / 2, cw - GAP, ch - GAP
        s = math.sqrt(iw * ih / n)
        cols = max(1, int(round(iw / s))); rows = math.ceil(n / cols)
        while cols * rows < n: cols += 1; rows = math.ceil(n / cols)
        sx, sy = iw / cols, ih / rows
        for k, i in enumerate(idx):
            xs[i] = ix + (k % cols + 0.5) * sx; ys[i] = iy + (k // cols + 0.5) * sy
        groups.append({"code": fam, "name": FAMILIES.get(fam, fam), "x": round(cx, 1), "y": round(cy, 1), "w": round(cw, 1), "h": round(ch, 1), "n": int(n), "pitch": round(min(sx, sy), 2)})
    yr = ft.disrupt_year_p50.to_numpy(dtype=float)
    yr = np.where(np.isfinite(yr), yr, 9999.0)
    share = ft.ai_share.fillna(0.45).clip(0, 1).to_numpy()
    feat = (ft.onet_soc_code == FEATURE).to_numpy()
    fx, fy = xs[feat], ys[feat]
    out = {"n": int(len(ft)), "x": [round(v, 1) for v in xs], "y": [round(v, 1) for v in ys],
           "year": [round(v, 2) for v in yr], "share": [round(v, 3) for v in share], "feature": [int(i) for i in np.flatnonzero(feat)],
           "feature_bbox": [round(fx.min(), 1), round(fy.min(), 1), round(fx.max(), 1), round(fy.max(), 1)], "feature_name": "Accountants and Auditors",
           "groups": groups, "source": "forecast_tasks.parquet: disrupt_year_p50 (Singapore, central case) and ai_share per O*NET task"}
    (Path(__file__).parent / "data" / "field.json").write_text(json.dumps(out, separators=(",", ":")))
    lit35 = float(((yr <= 2035.99)).mean())
    print(len(ft), "tasks;", len(groups), "families; feature tasks", int(feat.sum()), "bbox", out["feature_bbox"], "; share of tasks AI could be in by end-2035:", round(lit35, 3),
          "; median year", float(np.median(yr[yr < 9999])), "; never within horizon", int((yr >= 9999).sum()))
    print([(g["name"], g["n"], g["pitch"]) for g in groups][:8])


if __name__ == "__main__":
    main()
