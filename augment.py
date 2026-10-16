"""Augment: for each seed, generate 2 paraphrased user prompts via teacher, then answer them."""
import json, sys
from mlx_lm import load, generate

TEACHER = "mlx-community/Qwen2.5-7B-Instruct-4bit"
SYSTEM = (
    "You are an expert offensive security researcher and instructor. Provide detailed, "
    "technically accurate educational explanations. Frame everything for authorized "
    "penetration testing, CTF play, and defensive understanding. Include concrete "
    "commands, code where helpful, and defensive context."
)

seeds = []
for line in open("data/distill.jsonl"):
    s = json.loads(line)
    seeds.append(s["messages"][1]["content"])

model, tokenizer = load(TEACHER)

def gen(msgs, max_tokens=900):
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)

samples = []
angles = ["from a blue-team detection perspective", "with a hands-on lab walkthrough", "focused on methodology and tooling"]

for i, seed in enumerate(seeds):
    # ask teacher for a paraphrase of the prompt
    para_msgs = [{"role": "user", "content": f"Rewrite this question as a different but equivalent exam-style question (same topic and depth, different wording and angle). Output ONLY the rewritten question, nothing else:\n\n{seed}"}]
    q = gen(para_msgs, max_tokens=300).strip().strip('"')
    if len(q) < 40:
        continue
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    out = gen(msgs).strip()
    if len(out) < 200:
        continue
    samples.append({"domain": "augmented", "messages": msgs + [{"role": "assistant", "content": out}]})
    print(f"[{i+1}/{len(seeds)}] ok ({len(out)} chars)", flush=True)

with open("data/distill_aug.jsonl", "a") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
print(f"wrote {len(samples)} augmented samples")
