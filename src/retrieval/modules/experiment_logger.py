import csv
import os
from datetime import datetime

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/experiment_log.csv"))

def compute_avg_latency(results):
    latencies = [r.get("latency_sec", 0.0) for r in results]
    return round(sum(latencies) / len(latencies), 4) if latencies else 0.0

def log_experiment(
    experiment_name: str,
    mode: str,
    top_k: int,
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
            "timestamp", "experiment_name", "mode", "top_k",
            "P@K", "R@K", "F1@K", "MRR",
            "elapsed_time", "avg_latency",
            "hybrid_alpha", "dense_model", "notes"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "mode": mode,
            "top_k": top_k,
            "P@K": metrics.get("P@K", 0),
            "R@K": metrics.get("R@K", 0),
            "F1@K": metrics.get("F1@K", 0),
            "MRR": metrics.get("MRR", 0),
            "elapsed_time": round(elapsed_time, 4),
            "avg_latency": avg_latency,
            "hybrid_alpha": hybrid_alpha if mode == "Hybrid" else "",
            "dense_model": dense_model if mode == "Dense" else "",
            "notes": notes
        })
