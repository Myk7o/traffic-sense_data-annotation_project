import os, csv, io
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file, render_template
from datetime import datetime

app = Flask(__name__)

DATASET   = "path"
GTFINE    = "path"
CSV_PATH        = "path/inspection_results.csv"
MISSED_CSV_PATH = "path/missed_instances.csv"

INST_CLASS = {24: "person", 25: "cyclist", 26: "car", 32: "cyclist", 33: "cyclist"}

VALID_QUALITIES = [
    "fit",
    "under-annotated",
    "over-annotated",
    "mixed",
    "skip",
]

_images = None
_inst_cache = {}

# ── helpers ───────────────────────────────────────────────────────────────────

def image_index():
    global _images
    if _images is not None:
        return _images
    out = []
    for split in ["train", "val"]:
        root = os.path.join(DATASET, split, "masks")
        if not os.path.exists(root):
            continue
        for city in sorted(os.listdir(root)):
            cp = os.path.join(root, city)
            if not os.path.isdir(cp):
                continue
            for f in sorted(os.listdir(cp)):
                if not f.startswith(".") and f.endswith("_mask.png"):
                    out.append({"split": split, "city": city, "base": f[:-9]})
    _images = out
    return out

def load_inst(split, city, base):
    key = (split, city, base)
    if key not in _inst_cache:
        path = os.path.join(GTFINE, split, city, f"{base}_gtFine_instanceIds.png")
        if not os.path.exists(path):
            _inst_cache[key] = ([], None)
        else:
            arr = np.array(Image.open(path)).astype(np.int32)
            insts = []
            for uid in sorted(map(int, np.unique(arr))):
                if uid < 1000:
                    continue
                lid = uid // 1000
                if lid not in INST_CLASS:
                    continue
                insts.append({"id": uid, "label_id": lid,
                              "class": INST_CLASS[lid],
                              "pixels": int(np.sum(arr == uid))})
            insts.sort(key=lambda x: -x["pixels"])
            _inst_cache[key] = (insts, arr)
    return _inst_cache[key]

def load_reviews():
    rv = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rv[(row["split"], row["city"], row["base"], row["instance_id"])] = row
    return rv

def make_highlight(arr, iid, px=4):
    mask = (arr == iid)
    d = mask.copy()
    for _ in range(px):
        d = d | np.roll(d,1,0) | np.roll(d,-1,0) | np.roll(d,1,1) | np.roll(d,-1,1)
    outline = d & ~mask
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[mask]    = [255, 255,   0,  60]
    rgba[outline] = [255, 210,   0, 255]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    buf.seek(0)
    return buf

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/images")
def api_images():
    return jsonify(image_index())

@app.route("/api/image/<split>/<city>/<base>")
def api_image(split, city, base):
    insts, _ = load_inst(split, city, base)
    rv = load_reviews()
    result = []
    for i in insts:
        e = dict(i)
        k = (split, city, base, str(i["id"]))
        e["reviewed"] = k in rv
        if k in rv:
            e["review"] = rv[k]
        result.append(e)
    return jsonify({"split": split, "city": city, "base": base, "instances": result})

@app.route("/serve/rgb/<split>/<city>/<base>")
def serve_rgb(split, city, base):
    return send_file(os.path.join(DATASET, split, "images", city, f"{base}_leftImg8bit.png"))

@app.route("/serve/mask/<split>/<city>/<base>")
def serve_mask(split, city, base):
    return send_file(os.path.join(DATASET, split, "colored_masks", city, f"{base}_colored.png"))

@app.route("/serve/highlight/<split>/<city>/<base>/<int:iid>")
def serve_highlight(split, city, base, iid):
    _, arr = load_inst(split, city, base)
    if arr is None:
        return "Not found", 404
    return send_file(make_highlight(arr, iid), mimetype="image/png",
                     as_attachment=False, attachment_filename="hl.png")

@app.route("/api/click/<split>/<city>/<base>")
def api_click(split, city, base):
    x = int(float(request.args.get("x", 0)))
    y = int(float(request.args.get("y", 0)))
    _, arr = load_inst(split, city, base)
    if arr is None:
        return jsonify({"instance_id": None, "class": "background"})
    h, w = arr.shape
    uid = int(arr[max(0,min(y,h-1)), max(0,min(x,w-1))])
    lid = uid // 1000 if uid >= 1000 else uid
    return jsonify({"instance_id": uid, "class": INST_CLASS.get(lid, "background")})

@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.json
    data["timestamp"] = datetime.now().isoformat()
    fields = ["split","city","base","instance_id","class","quality_label",
              "parts_affected","notes","reviewer","timestamp"]
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(data)
    return jsonify({"ok": True})

@app.route("/api/save_missed", methods=["POST"])
def api_save_missed():
    data = request.json
    data["timestamp"] = datetime.now().isoformat()
    fields = ["split", "city", "base", "description", "reviewer", "timestamp"]
    new_file = not os.path.exists(MISSED_CSV_PATH)
    with open(MISSED_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(data)
    return jsonify({"ok": True})

@app.route("/api/progress")
def api_progress():
    rv = load_reviews()
    return jsonify({"reviewed": len(rv), "images": len(image_index())})

if __name__ == "__main__":
    image_index()
    print(f"Loaded {len(_images)} images  →  http://localhost:5001")
    app.run(debug=True, port=5001)
