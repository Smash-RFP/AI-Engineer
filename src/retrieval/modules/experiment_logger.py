import csv
import os
from datetime import datetime

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "/home/gcp-JeOn/Smash-RFP/src/retrieval/results/experiment_log.csv"))

def compute_avg_latency(results):
    latencies = [r.get("latency_sec", 0.0) for r in results]
    return round(sum(latencies) / len(latencies), 4) if latencies else 0.0


def log_2stage_experiment(
    experiment_name: str,
    mode: str,
    k: int,
    metrics: dict,
    elapsed_time: float,
    avg_latency: float,
    hybrid_alpha: float = None,
    dense_model: str = None,
    notes: str = ""
):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "experiment_name", "mode", "k",
            "Doc_P@K", "Doc_R@K", "Chunk_P@K", "Chunk_R@K",
            "elapsed_time", "avg_latency",
            "hybrid_alpha", "dense_model", "notes"
        ])
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "experiment_name": experiment_name,
            "mode": mode,
            "k": k,
            "Doc_P@K": metrics["Doc_P@K"][k],
            "Doc_R@K": metrics["Doc_R@K"][k],
            "Chunk_P@K": metrics["Chunk_P@K"][k],
            "Chunk_R@K": metrics["Chunk_R@K"][k],
            "elapsed_time": round(elapsed_time, 4),
            "avg_latency": avg_latency,
            "hybrid_alpha": hybrid_alpha if mode == "Hybrid" else "",
            "dense_model": dense_model if mode == "Dense" else "",
            "notes": notes
        })
