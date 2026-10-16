from huggingface_hub import HfApi, snapshot_download

matches = [m.id for m in HfApi().list_models(search="Qwen2.5-0.5B-Instruct mlx", limit=10)]
print("matches:", matches)

target = None
for m in matches:
    if "mlx-community" in m and "4bit" in m:
        target = m
        break
if target is None:
    for m in matches:
        if "mlx-community" in m:
            target = m
            break

print("downloading:", target)
p = snapshot_download(target)
print("path:", p)
