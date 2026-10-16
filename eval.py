import time
from mlx_lm import load, generate

PROMPTS = [
    "Explain how Kerberoasting works and how to detect it.",
    "What is SSRF and how would you test for it in an authorized pentest?",
    "How does tcache poisoning work in glibc heap exploitation?",
    "Write a Python script to enumerate subdomains from crt.sh with rate limiting.",
]

SYSTEM = (
    "You are an expert offensive security researcher and instructor. Provide detailed, "
    "technically accurate educational explanations. Frame everything for authorized "
    "penetration testing, CTF play, and defensive understanding. Include concrete "
    "commands, code where helpful, and defensive context."
)

BASE = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
FT = ("mlx-community/Qwen2.5-0.5B-Instruct-bf16", "adapters")

for label, cfg in [("BASE", (BASE, None)), ("FINE-TUNED", FT)]:
    model, tok = load(cfg[0], adapter_path=cfg[1])
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    for p in PROMPTS:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        out = generate(model, tok, prompt=text, max_tokens=400, verbose=False)
        dt = time.time() - t0
        print(f"\n>>> {p}\n[{dt:.1f}s, {len(out)} chars]\n{out[:700]}", flush=True)
    del model, tok
