# SentryBench

**SentryBench** is an open-source, defense-first framework for **backdoor evaluation, reproducible benchmarking, and modular defenses** for modern LLM workflows (e.g., **LoRA/QLoRA adapters**, merging, instruction tuning).

---

## Why SentryBench

Backdoor research often suffers from fragmentation: different datasets, triggers, training recipes, and metrics make results hard to compare.  
SentryBench provides a **standard experiment contract**:

- **Unified data schema** (JSONL) + standardized task/benchmark definition
- **Modular defenses** with a consistent interface (`fit/apply/evaluate`)
- **Reproducible pipelines** (seeds, configs, artifacts, hashes)
- **One-command reports** (ASR/utility/stealth/robustness + CI-friendly outputs)

---

## Scope & Principles

### What this repo is
- A **benchmark harness** + **defense toolkit** for backdoor-related research and engineering.
- A **reproducibility layer** for adapter-based and instruction-tuned settings.

### What this repo is NOT
- Not a “one-click attack kit” for malicious use.
- Not a collection of arbitrary scripts without standardized evaluation.

---

## Responsible Use (Dual-Use Notice)

Backdoor techniques are dual-use. SentryBench is designed for **research, defense, and evaluation**.

- **Defense & evaluation come first.**
- Attack modules (if included) are intended for **controlled, research-oriented reproduction** and may be **restricted by default**.
- Do not use this project to compromise systems or deploy harmful behavior.

See: [Responsible Use Policy](#responsible-use-policy).

---

## Key Features (Planned / Initial)

- **Config-driven pipelines**: run experiments via YAML configs
- **Adapter-centric workflow**: LoRA/QLoRA training, merge/unmerge, delta extraction
- **Metrics**:
  - Attack Success Rate (ASR)
  - Utility (task accuracy / helpfulness proxies)
  - Stealth (behavioral drift / perplexity shift / representation shift; extensible)
  - Robustness (trigger variants, decoding variants, prompt templates)
- **Standard reports**:
  - JSON summary for CI
  - Markdown/HTML report for human review
- **Extensible plugin system**:
  - Add a new defense/metric/benchmark via a minimal interface

---

## Quick Start (Planned)

> This section will be updated once the first runnable skeleton is merged.

```bash
# 1) Install
pip install -e .

# 2) Run an evaluation (example)
sentrybench run -c configs/example_lora_eval.yaml

# 3) Generate a report
sentrybench report -i runs/2026-02-08_01/ -o report.md
