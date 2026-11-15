"""
ceiling.py - přibližný "lidský strop" (krok 4).

U pozorování s aspoň dvěma anotátory vezme sadu jednoho anotátora jako "predikci"
a sady ostatních jako ground truth. Výsledné PQ říká, jak moc se lidé shodují mezi sebou.

Spuštění:  python ceiling.py            (potřebuje work/gt_all.csv z `pq_eval.py gt`)
"""
from pathlib import Path
import pandas as pd

from pq_eval import score


def load_gt(path=None):
    if path is None:
        path = str(Path(__file__).resolve().parent / "work" / "gt_all.csv")
    gt = pd.read_csv(path)
    ai = gt["filament_id"].str.split("_", n=1).str[0]            # "<anotátor>-<snímek>"
    gt["annotator"] = ai.str.split("-", n=1).str[0]
    gt["obs"] = ai.str.split("-", n=1).str[1]
    n_ann = gt.groupby("obs")["annotator"].nunique()
    gt = gt[gt["obs"].isin(n_ann[n_ann >= 2].index)].copy()      # jen opakovaně anotované
    gt["rank"] = gt.groupby("obs")["annotator"].transform(
        lambda s: pd.Series(pd.factorize(s, sort=True)[0], index=s.index)
    )
    return gt


def split(gt, r, limit=None):
    """Predikce = r-tý anotátor (podle pořadí ID), GT = všichni ostatní u téhož snímku."""
    pred = gt[gt["rank"] == r]
    obs = sorted(pred["obs"].unique())[:limit]
    pred = pred[pred["obs"].isin(obs)].copy()
    ref = gt[(gt["rank"] != r) & gt["obs"].isin(obs)]
    pred["filament_id"] = pred["obs"] + "_" + (pred.groupby("obs").cumcount() + 1).astype(str)
    cols = ["filament_id", "segmentation_rle"]
    return ref[cols], pred[cols]


if __name__ == "__main__":
    gt = load_gt()
    print(f"pozorování s 2+ anotátory: {gt['obs'].nunique()}")
    for r in (0, 1, 2):
        ref, pred = split(gt, r)
        if pred.empty:
            continue
        res = score(ref, pred)
        print(f"predikuje anotátor č. {r}:",
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
