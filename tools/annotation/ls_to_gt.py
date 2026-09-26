# ls_to_gt.py — Label Studio JSON export -> ground_truth.json
# usage: python ls_to_gt.py export_events_A.json gt_A.json [--user ID_или_email] [--all]
import argparse
import json

FPS_DEFAULT = 30000 / 1001

def _user_ok(a, user):
    if user is None:
        return True
    cb = a.get("completed_by")
    if isinstance(cb, dict):                 # в части экспортов completed_by — объект
        return str(user) in (str(cb.get("id")), str(cb.get("email")))
    return str(cb) == str(user)

def ls_to_gt(export_path, out_path, user=None, take_all=False):
    with open(export_path, encoding="utf-8") as fh:
        tasks = json.load(fh)
    gt = {}
    for t in tasks:
        d = t["data"]
        vid = d["orig"]                          # имя ОРИГИНАЛА (ключ GT), не прокси
        fps = float(d.get("fps", FPS_DEFAULT))
        dur = float(d["duration"])
        e = gt.setdefault(vid, {"duration": dur, "fps": round(fps, 2), "events": []})
        anns = [a for a in t.get("annotations", [])
                if not a.get("was_cancelled") and _user_ok(a, user)]
        if anns and not take_all:                # по умолчанию одна (последняя) аннотация на задачу
            anns = [max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")]
        for a in anns:
            for r in a.get("result", []):
                if r.get("type") != "timelinelabels":
                    continue
                labs = r["value"].get("timelinelabels") or []
                for rg in r["value"].get("ranges", []) if labs else []:
                    s = (rg["start"] - 1) / fps  # LS: 1-based, включительно
                    en = min(rg["end"] / fps, dur)
                    e["events"].append([round(s, 3), round(en, 3), labs[0]])
    for v in gt.values():
        v["events"].sort()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=1, ensure_ascii=False)
    return gt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("out")
    ap.add_argument("--user", help="id или e-mail аннотатора (поле completed_by)")
    ap.add_argument("--all", action="store_true", help="взять все аннотации задачи, а не последнюю")
    args = ap.parse_args()
    gt = ls_to_gt(args.export, args.out, args.user, args.all)
    print({k: len(v["events"]) for k, v in gt.items()})
