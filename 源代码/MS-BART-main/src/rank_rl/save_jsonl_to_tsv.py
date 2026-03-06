import pandas as pd

import sys
sys.path.append(".")
from src.utils import read_jsonl

data = read_jsonl("logs/datasets/MassSpecGym-ALL/train/candidates/MassSpecGym-ALL_fps_selfies_threshold_0.2.jsonl")

final_data = []
for d in data:
    candidates = d["candidates"]
    candidates = list(map(lambda x: x[0], candidates))
    candidates_str = ",".join(candidates)
    final_data.append({
        "fps": d["fps"],
        "label": d["label"],
        "candidates": candidates_str
    })

df = pd.DataFrame(final_data)

df.to_csv("logs/datasets/MassSpecGym-ALL/train/candidates/MassSpecGym-ALL_fps_selfies_threshold_0.2.tsv", sep='\t', index=False)
