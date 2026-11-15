"""
pq_eval.py - lokální validace pro Solar Filament Segmentation Challenge 2026.

Počítá Panoptic Quality stejně jako oficiální Self_Evaluation_Notebook:
  * jeden řádek GT = jeden filament, filament_id = "<anotátor>-<snímek>_<n>"
  * každá dvojice (anotátor, snímek) se hodnotí zvlášť proti stejným predikcím snímku
  * dvojice GT/predikce je TP, když IoU > 0.5 (bez 1:1 párování, jako oficiální kód)
  * PQ = sum(IoU TP) / (TP + 0.5*FP + 0.5*FN), sečteno přes všechny snímky najednou

Použití:
  python pq_eval.py gt    --json train.json --out gt_all.csv
  python pq_eval.py folds --json train.json --k 5 --seed 2026 --out folds.csv
  python pq_eval.py score --gt gt_all.csv --pred pred.csv [--folds folds.csv --fold 0]

Vyžaduje pandas, numpy a (pro gt/score) pycocotools.
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd

IOU_THRESHOLD = 0.5
SIZE = (2048, 2048)


# ---------------------------------------------------------------- jádro metriky
def pq_from_matrices(matrices):
    """matrices: seznam polí (n_gt, n_pred), jedno na (anotátor, snímek).

    Logika odpovídá oficiálnímu get_pq_score. Navíc vrací diagnostiku
    gt_multi_hit / pred_multi_hit: kolik GT (predikcí) má víc než jeden zásah
    s IoU > 0.5. Oficiální kód počítá každý takový zásah jako samostatné TP,
    takže u překrývajících se predikcí se skóre uměle mění. Měj to na nule.
    """
    tp_iou, fp, fn, gt_multi, pred_multi = [], 0, 0, 0, 0
    for m in matrices:
        n_gt, n_pred = m.shape
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        hit = m > IOU_THRESHOLD
        tp_iou.extend(m[hit].tolist())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
        gt_multi += int((hit.sum(axis=1) > 1).sum())
        pred_multi += int((hit.sum(axis=0) > 1).sum())
    tp = len(tp_iou)
    denom = tp + 0.5 * fp + 0.5 * fn
    total = float(sum(tp_iou))
    return {
        "pq": total / denom if denom > 0 else 0.0,
        "sq": total / tp if tp else 0.0,
        "rq": tp / denom if denom > 0 else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "gt_multi_hit": gt_multi, "pred_multi_hit": pred_multi,
    }


# ------------------------------------------------------------------- RLE a IoU
def _rles(strings):
    return [{"size": list(SIZE), "counts": str(s).encode("ascii")} for s in strings]


def iou_matrix(gt_strings, pred_strings):
    """IoU (n_gt, n_pred) přímo z komprimovaných COCO RLE, bez dekódování do 2048x2048."""
    from pycocotools import mask as mu
    if not gt_strings or not pred_strings:
        return np.zeros((len(gt_strings), len(pred_strings)))
    iou = mu.iou(_rles(pred_strings), _rles(gt_strings), [0] * len(gt_strings))
    return np.asarray(iou, dtype=np.float64).T


def score(gt_df, pred_df):
    """Oba DataFrame mají sloupce filament_id, segmentation_rle."""
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]            # anotátor-snímek
    pred_img = pred_df["filament_id"].str.split("_", n=1).str[0]       # snímek
    pred_by_img = pred_df.groupby(pred_img)["segmentation_rle"].apply(list).to_dict()
    matrices, missing = [], set()
    for ai, rows in gt_df.groupby(gt_ai, sort=False):
        img = ai.split("-", 1)[1]
        if img not in pred_by_img:
            missing.add(img)
        matrices.append(iou_matrix(rows["segmentation_rle"].tolist(), pred_by_img.get(img, [])))
    out = pq_from_matrices(matrices)
    out["annotator_images"] = len(matrices)
    out["images_without_prediction"] = len(missing)
    return out


# ------------------------------------------------------------ GT a foldy z JSONu
def build_gt(json_path):
    """Všichni anotátoři, formát stejný jako submission (+ category_id)."""
    from pycocotools import mask as mu
    coco = json.load(open(json_path))
    images = {im["id"]: im for im in coco["images"]}
    counter, rows = defaultdict(int), []
    for ann in coco["annotations"]:
        im = images[ann["image_id"]]
        rle = mu.merge(mu.frPyObjects(ann["segmentation"], im["height"], im["width"]))
        counter[im["id"]] += 1
        rows.append({
            "filament_id": f'{im["id"]}_{counter[im["id"]]}',
            "segmentation_rle": rle["counts"].decode("ascii"),
            "category_id": ann["category_id"],
        })
    return pd.DataFrame(rows)


def make_folds(json_path, k=5, seed=2026):
    """Foldy po fyzických pozorováních (všichni anotátoři jednoho snímku spolu),
    vyvážené podle observatoře (poslední 2 znaky ID, např. 'Mh')."""
    coco = json.load(open(json_path))
    observations = sorted({im["id"].split("-", 1)[1] for im in coco["images"]})
    by_site = defaultdict(list)
    for o in observations:
        by_site[o[-2:]].append(o)
    rng, fold, i = random.Random(seed), {}, 0
    for site in sorted(by_site):
        items = by_site[site]
        rng.shuffle(items)
        for o in items:
            fold[o] = i % k
            i += 1
    return pd.DataFrame({"observation": list(fold), "fold": list(fold.values())})


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("gt")
    a.add_argument("--json", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("folds")
    b.add_argument("--json", required=True)
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--seed", type=int, default=2026)
    b.add_argument("--out", required=True)
    c = sub.add_parser("score")
    c.add_argument("--gt", required=True)
    c.add_argument("--pred", required=True)
    c.add_argument("--folds")
    c.add_argument("--fold", type=int)
    args = ap.parse_args()

    if args.cmd == "gt":
        build_gt(args.json).to_csv(args.out, index=False)
    elif args.cmd == "folds":
        make_folds(args.json, args.k, args.seed).to_csv(args.out, index=False)
    else:
        gt, pred = pd.read_csv(args.gt), pd.read_csv(args.pred)
        if args.folds is not None:
            f = pd.read_csv(args.folds)
            keep = set(f.loc[f["fold"] == args.fold, "observation"])
            obs = gt["filament_id"].str.split("_", n=1).str[0].str.split("-", n=1).str[1]
            gt = gt[obs.isin(keep)]
        res = score(gt, pred)
        for key, val in res.items():
            print(f"{key:26s} {val:.4f}" if isinstance(val, float) else f"{key:26s} {val}")


if __name__ == "__main__":
    main()
