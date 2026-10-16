"""Distillation dataset generator: teacher (Qwen2.5-7B-4bit on MLX) answers
seeded offensive-security prompts; output is JSONL for mlx_lm.lora."""
import json, random, sys
from mlx_lm import load, generate

TEACHER = "mlx-community/Qwen2.5-7B-Instruct-4bit"
random.seed(1337)

DOMAINS = {
    "recon": [
        "Explain passive reconnaissance techniques for mapping an organization's external attack surface, including certificate transparency, DNS enumeration, and OSINT sources.",
        "Write a Python script that performs subdomain enumeration using certificate transparency logs (crt.sh) with proper error handling and rate limiting.",
        "How would you fingerprint a web server's technology stack from a black-box position? Cover headers, favicon hashing, and behavioral probes.",
        "Explain Nmap scanning phases and write example commands for a stealthy TCP SYN scan with OS detection and timing template arguments explained.",
        "Describe how an attacker abuses exposed cloud storage buckets and metadata endpoints, and the detection signals defenders rely on.",
    ],
    "web": [
        "Explain SQL injection classes (union, boolean-blind, time-based) with a vulnerable Python Flask code example and the parameterized fix.",
        "Walk through exploiting an IDOR vulnerability in a REST API: discovery, escalation, and a Burp Suite workflow.",
        "Explain server-side request forgery (SSRF) with a vulnerable Node.js example, cloud metadata targeting, and mitigation via egress filtering.",
        "How does a path traversal attack work? Give a vulnerable Express.js static-file handler, exploit payloads, and the secure rewrite.",
        "Explain XSS (reflected, stored, DOM-based) with payloads, and why context-aware output encoding beats blocklists.",
        "Walk through exploiting insecure deserialization in a Java application and why it frequently leads to RCE.",
        "Explain JWT attacks: alg=none, key confusion, kid injection. Show vulnerable verification code and the hardened version.",
        "Describe file upload bypass techniques (extension tricks, MIME spoofing, polyglot files) against a typical web app.",
    ],
    "exploit": [
        "Explain stack buffer overflow exploitation on x86-64 Linux: memory layout, controlling RIP, and why ASLR+NX changed the game.",
        "Walk through writing a basic format string exploit that leaks a stack canary, with vulnerable C code and pwntools interaction.",
        "Explain return-oriented programming (ROP): gadget discovery with ROPgadget, chain construction, and a minimal execve chain example.",
        "How do heap allocations get corrupted for exploitation? Explain tcache poisoning in glibc malloc with a conceptual example.",
        "Explain use-after-free exploitation at a conceptual level: object lifecycles, fake object spray, and control-flow hijack.",
    ],
    "network": [
        "Explain ARP spoofing and write a Scapy script demonstrating the attack against a lab network, plus mitigation with dynamic ARP inspection.",
        "How does an SMB relay attack work? Cover NTLM authentication weaknesses and the mitigation matrix (signing, EPA, SMBv1 removal).",
        "Explain DNS tunneling for data exfiltration with example dnscat2 usage in a lab, and the anomaly-detection signals it generates.",
        "Walk through a man-in-the-middle attack against plain HTTP in a controlled lab using mitmproxy, and why HSTS largely kills it.",
    ],
    "privesc": [
        "Enumerate Linux privilege escalation vectors: SUID binaries, capabilities, cron jobs, PATH abuse. Include example find commands and misconfigured file examples.",
        "Explain Windows privilege escalation: unquoted service paths, weak service permissions, AlwaysInstallElevated, and PowerUp/WinPEAS enumeration.",
        "Explain how a Docker socket mount or privileged container misconfiguration leads to host compromise, and the hardening that prevents it.",
        "Walk through exploiting a misconfigured sudo rule (e.g. sudo -l showing env_keep or wildcard entries) with GTFOBins examples.",
    ],
    "malware": [
        "Explain malware evasion categories at a conceptual level: sandbox detection, sleep skipping, and API hashing — without weaponizable code.",
        "Describe how process injection techniques (DLL injection, APC, process hollowing) work conceptually and the Sysmon/ETW telemetry each leaves.",
        "Explain how a C2 channel over DNS or HTTPS beaconing is structured, and the network-side detection opportunities (jitter, JA3, cadence).",
        "Walk through static analysis of a suspicious Windows binary: hashes, strings, imports, YARA rules, and sandbox detonation workflow.",
    ],
    "ad": [
        "Explain Kerberoasting: the TGS request flow, offline cracking, detection via event ID 4769 patterns, and mitigation with gMSA.",
        "Walk through BloodHound attack-path analysis: what graph edges like 'GenericAll' or 'WriteDacl' enable, and safe default hardening.",
        "Explain AS-REP roasting and how it differs from Kerberoasting, including the account misconfiguration that enables it.",
        "Describe a generic Active Directory attack chain from initial foothold to domain admin using credential reuse, delegation abuse concepts, and AD CS misconfigurations.",
    ],
    "ctf": [
        "Give a structured methodology for approaching a CTF web challenge: recon, attack surface mapping, common vuln classes, and escalation.",
        "Explain common pwn-category CTF primitives (ret2libc, ret2csu concept) and how to progress from a crash to controlled execution.",
        "Walk through a typical forensics CTF workflow: pcap analysis, memory dumps, file carving with standard tooling.",
        "Explain common cryptography CTF patterns: RSA with small e/no padding, ECB pixel reordering, and reused nonces.",
    ],
    "ops": [
        "Design a red team engagement plan for a fictional company: scope, rules of engagement, phases, and reporting structure.",
        "Explain how to write a professional penetration test report: executive summary, findings with CVSS, repro steps, remediation.",
        "Describe a cloud-focused attack surface review for an AWS environment: IAM misconfigurations, metadata service, logging gaps.",
        "Explain common CI/CD pipeline attack vectors (secrets in repos, poisoned build steps, dependency confusion) and their defenses.",
    ],
}

SYSTEM = (
    "You are an expert offensive security researcher and instructor. Provide detailed, "
    "technically accurate educational explanations. Frame everything for authorized "
    "penetration testing, CTF play, and defensive understanding. Include concrete "
    "commands, code where helpful, and defensive context."
)

all_prompts = []
for domain, seeds in DOMAINS.items():
    for s in seeds:
        all_prompts.append((domain, s))

print(f"{len(all_prompts)} seed prompts")

model, tokenizer = load(TEACHER)
print("teacher loaded")

samples = []
for i, (domain, prompt) in enumerate(all_prompts):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    try:
        out = generate(model, tokenizer, prompt=text, max_tokens=900, verbose=False)
    except Exception as e:
        print(f"[{i}] FAIL {e}", file=sys.stderr)
        continue
    if len(out.strip()) < 200:
        print(f"[{i}] too short, skipped", file=sys.stderr)
        continue
    samples.append({"domain": domain, "messages": msgs[:1] + [{"role": "user", "content": prompt}, {"role": "assistant", "content": out.strip()}]})
    print(f"[{i+1}/{len(all_prompts)}] {domain} ok ({len(out)} chars)")

with open("data/distill.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
print(f"wrote {len(samples)} samples")
