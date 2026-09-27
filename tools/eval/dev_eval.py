# dev_eval.py — score our rule detectors against the team's labels, class by class.
#
#   python tools/eval/dev_eval.py [--classes jaywalking failure_to_yield ...] [--errors jaywalking]
#
# Ground truth: labels/dev/sources.json lists, per video, the converted team labels
# (tools/annotation/cvat_to_gt.py output) and which classes that annotator covered
# ("all" or a list). A class is scored only on videos where it was covered, so a video
# labelled for jaywalking only does not turn our other predictions into false positives.
# Metric: exactly the official one (evaluate.py match_segments / prf): TP/FP/FN pooled over
# videos, F1 at tIoU 0.3/0.5/0.7, mean over the thresholds.
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from evaluate import OFFICIAL_CLASSES, TIOU_THRESHOLDS, match_segments, prf  # noqa: E402
from westcv import rules  # noqa: E402
from westcv.scene import Scene  # noqa: E402

DETECTORS = rules.RULES


def load_gt():
    src = json.loads((ROOT / "labels/dev/sources.json").read_text(encoding="utf-8"))
    gt = {}
    for stem, s in src.items():
        entry = next(iter(json.loads((ROOT / s["file"]).read_text(encoding="utf-8")).values()))
        covered = set(OFFICIAL_CLASSES) if s["classes"] == "all" else set(s["classes"])
        gt[stem] = {"events": entry["events"], "covered": covered, "who": s["who"]}
    return gt


def predict(stem, classes, tag=""):
    sc = Scene.from_cache(stem, ROOT, tag)
    return sorted([round(float(a), 3), round(float(b), 3), c] for c in classes for a, b in DETECTORS[c](sc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(DETECTORS))
    ap.add_argument("--errors", nargs="*", default=[], help="print unmatched GT / predictions for these classes")
    ap.add_argument("--save", default=str(ROOT / "cache/dev_pred.json"), help="predictions (not the labels folder)")
    ap.add_argument("--tag", default="_w1280", help="cache variant: _w1280 = the pipeline's 1280 px, '' = 1920 px")
    args = ap.parse_args()
    gt = load_gt()
    pred = {stem: predict(stem, args.classes, args.tag) for stem in gt}
    Path(args.save).write_text(json.dumps(pred, indent=1), encoding="utf-8")

    print(f"{'class':22s} {'videos':>6s} {'GT':>4s} {'pred':>5s}  F1@0.3  F1@0.5  F1@0.7   mean   TP/FP/FN@0.5")
    for c in args.classes:
        vids = [v for v in gt if c in gt[v]["covered"]]
        g = {v: [(s, e) for s, e, lab in gt[v]["events"] if lab == c] for v in vids}
        p = {v: [(s, e) for s, e, lab in pred[v] if lab == c] for v in vids}
        f1s, at5 = [], None
        for thr in TIOU_THRESHOLDS:
            tp = fp = fn = 0
            for v in vids:
                a, b, d = match_segments(g[v], p[v], thr)
                tp, fp, fn = tp + a, fp + b, fn + d
            r = prf(tp, fp, fn)
            f1s.append(r["f1"])
            if thr == 0.5:
                at5 = (tp, fp, fn)
        n_g, n_p = sum(map(len, g.values())), sum(map(len, p.values()))
        print(f"{c:22s} {len(vids):6d} {n_g:4d} {n_p:5d}  {f1s[0]:6.3f}  {f1s[1]:6.3f}  {f1s[2]:6.3f}  {sum(f1s) / 3:6.3f}   "
              f"{at5[0]}/{at5[1]}/{at5[2]}")
        if c in args.errors:
            from evaluate import tiou
            for v in vids:
                for s in g[v]:
                    best = max((tiou(s, q) for q in p[v]), default=0)
                    if best < 0.5:
                        print(f"    FN {v} {s[0]:7.2f}-{s[1]:7.2f}  best tIoU {best:.2f}")
                for q in p[v]:
                    best = max((tiou(s, q) for s in g[v]), default=0)
                    if best < 0.5:
                        print(f"    FP {v} {q[0]:7.2f}-{q[1]:7.2f}  best tIoU {best:.2f}")


if __name__ == "__main__":
    main()
