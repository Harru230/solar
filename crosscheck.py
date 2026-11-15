"""
crosscheck.py - krok 5: porovná můj PQ s oficiálním Self_Evaluation_Notebook.

Příprava (jednou):
  pip install nbconvert matplotlib
  jupyter nbconvert --to script self-evaluation-notebook.ipynb --output self_eval
Soubor self_eval.py nech v kořeni repa, ale NEcommituj ho (kód organizátorů, přidej do .gitignore).

Spuštění:  python crosscheck.py
"""
from ceiling import load_gt, split
from pq_eval import score

import self_eval as official  # oficiální notebook převedený na .py

ref, pred = split(load_gt(), r=0, limit=15)          # 15 pozorování stačí
ref, pred = ref.reset_index(drop=True), pred.reset_index(drop=True)

official_pq = official.get_pq_score(official.get_overlap_df(ref, pred))
mine = score(ref, pred)["pq"]

print(f"oficiální PQ: {official_pq:.6f}")
print(f"můj PQ:       {mine:.6f}")
print("SHODA" if abs(official_pq - mine) < 1e-3 else "ROZDÍL - pošli mi obě čísla")
