"""
Fetch only the ES2002a rows from the HuggingFace AMI dataset
using the datasets-server HTTP API — no full parquet download needed.
"""
import sys, json, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = "https://datasets-server.huggingface.co"

def api_get(path):
    req = urllib.request.Request(
        BASE + path,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())

# Step 1: find available configs/splits
print("Checking available configs...")
try:
    info = api_get("/info?dataset=edinburghcstr%2Fami")
    configs = list(info.get("dataset_info", {}).keys())
    print("Configs:", configs[:5])
except Exception as e:
    print("info failed:", e)
    configs = ["ihm", "sdm", "mdm"]

# Step 2: search rows containing ES2002a
for config in configs[:3]:
    print(f"\nSearching config={config} for ES2002a...")
    for split in ["test", "validation", "train"]:
        try:
            # Use /search endpoint to find ES2002a rows
            params = urllib.parse.urlencode({
                "dataset": "edinburghcstr/ami",
                "config": config,
                "split": split,
                "query": "ES2002a",
            })
            result = api_get(f"/search?{params}")
            rows = result.get("rows", [])
            print(f"  split={split}: {len(rows)} rows match 'ES2002a'")
            if rows:
                print("  Sample row keys:", list(rows[0].get("row", {}).keys()))
                print("  Sample:", str(rows[0].get("row", {}))[:300])
                # Collect all ES2002a texts
                texts = []
                for row in rows:
                    r = row.get("row", {})
                    text = r.get("text", r.get("transcript", r.get("words", "")))
                    if text:
                        texts.append(str(text).strip())
                if texts:
                    ref = " ".join(texts)
                    print(f"\n  Total words: {len(ref.split())}")
                    print(f"  Preview: {ref[:400]}")
                    with open("ami_es2002a_ref.txt", "w", encoding="utf-8") as f:
                        f.write(ref)
                    print("  Saved to ami_es2002a_ref.txt")
                    sys.exit(0)
        except Exception as e:
            print(f"  split={split} failed: {e}")

print("\nCould not retrieve ES2002a via API.")
