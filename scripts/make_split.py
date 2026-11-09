import argparse, json, random, re, shutil
from collections import defaultdict
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def defect_of(name, label):
    if label == "good":
        return "none"
    parts = Path(name).stem.split("_")
    return parts[2] if len(parts) > 3 else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--src", default="data/quality_split")
    ap.add_argument("--dst", default="data/quality_split_ready")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        shutil.rmtree(dst)
    rng = random.Random(a.seed)
    meta, counts = [], defaultdict(int)
    for label in ("good", "bad"):
        files = sorted(p for p in (src / label).iterdir() if p.suffix.lower() in EXT)
        groups = defaultdict(list)
        for p in files:                                   # group variants of the same source image
            groups[re.sub(r"_v\d+$", "", p.stem)].append(p)
        keys = sorted(groups); rng.shuffle(keys)
        n = len(files); target = {"train": 0.70 * n, "validation": 0.15 * n}
        split, got = "train", 0
        assign = {}
        for k in keys:
            if split == "train" and got >= target["train"]:
                split, got = "validation", 0
            if split == "validation" and got >= target["validation"]:
                split, got = "test", 0
            assign[k] = split; got += len(groups[k])
        for k, ps in groups.items():
            for p in ps:
                out = dst / assign[k] / label; out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out / p.name); counts[(assign[k], label)] += 1
                meta.append({"filename": p.name, "split": assign[k], "quality_label": label,
                             "defect": defect_of(p.name, label)})
    with open(dst.parent / "split_metadata.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(m, ensure_ascii=False) for m in meta))
    for s in ("train", "validation", "test"):
        print(f"{s:<11}", {l: counts[(s, l)] for l in ("good", "bad")})
    print("written:", dst, "and", dst.parent / "split_metadata.jsonl")


if __name__ == "__main__":
    main()
