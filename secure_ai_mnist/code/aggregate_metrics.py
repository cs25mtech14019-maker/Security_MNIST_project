import os, json, csv

BASE_DIR = os.path.join(os.getcwd(), "secure_ai_mnist")
OUTPUT_JSON = os.path.join(BASE_DIR, "metrics_summary.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "metrics_summary.csv")

FILES = [
    ("baseline", os.path.join(BASE_DIR, "baseline_metrics.json")),
    ("fgsm", os.path.join(BASE_DIR, "fgsm_metrics.json")),
    ("advtrain", os.path.join(BASE_DIR, "advtrain_metrics.json")),
    ("poison", os.path.join(BASE_DIR, "poison_metrics.json")),
]

summary = {}
for label, path in FILES:
    if os.path.exists(path):
        try:
            with open(path) as f:
                summary[label] = json.load(f)
        except Exception as e:
            summary[label] = {"error": str(e)}

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)

# Flatten selective numeric metrics for CSV
rows = []
header = ["section", "metric", "value"]
for section, data in summary.items():
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (int, float)):
                rows.append([section, k, v])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("Aggregated metrics written:")
print(" -", OUTPUT_JSON)
print(" -", OUTPUT_CSV)
