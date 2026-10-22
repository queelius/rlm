# Links and prior art

## Our releases

Models (LoRA adapters for Qwen3-4B-Instruct-2507, Apache-2.0):
[rlm-qwen3-4b-sft-005](https://huggingface.co/queelius/rlm-qwen3-4b-sft-005) (taught warm start),
[rlm-qwen3-4b-rl-mixed-001](https://huggingface.co/queelius/rlm-qwen3-4b-rl-mixed-001) (best RL
controller). Datasets (CC-BY-4.0):
[rlm-teaching-eventlog](https://huggingface.co/datasets/queelius/rlm-teaching-eventlog),
[rlm-teaching-oolong-cells](https://huggingface.co/datasets/queelius/rlm-teaching-oolong-cells),
[rlm-oolong-split-and-verdicts](https://huggingface.co/datasets/queelius/rlm-oolong-split-and-verdicts).
Small caller (21 October): [rlm-qwen3-0.6b-onecell-caller](https://huggingface.co/queelius/rlm-qwen3-0.6b-onecell-caller)
(LoRA for Qwen3-0.6B that matches the 4B caller with the library) and its training set
[rlm-onecell-library-calls](https://huggingface.co/datasets/queelius/rlm-onecell-library-calls) (MIT).
Reproductions: [SPO on GSM8K](https://github.com/queelius/spo-gsm8k-reproduction) with its
[walkthrough](https://queelius.github.io/spo-gsm8k-reproduction/) and
[checkpoint](https://huggingface.co/queelius/spo-chain-rho1.1b-gsm8k-reproduction);
[R1-style Dr. GRPO and CURL](https://github.com/queelius/rl-reproductions). Archived September
notebook: [rlm-research](https://github.com/queelius/rlm-research). Prior study:
[structured-decomposition-benchmark](https://github.com/queelius/structured-decomposition-benchmark).

## Recursive language models

- Zhang, Kraska, Khattab. *Recursive Language Models*. arXiv:2512.24601 (NeurIPS 2026).
  [Paper](https://arxiv.org/abs/2512.24601), [blog](https://alexzhang13.github.io/blog/2025/rlm/),
  [code](https://github.com/alexzhang13/rlm), post-trained
  [rlm-qwen3-8b](https://huggingface.co/mit-oasys/rlm-qwen3-8b-v0.1).
- Prime Intellect, *Recursive Language Models: the paradigm of 2026* (January 2026):
  [post](https://www.primeintellect.ai/blog/rlm); RLM environments in verifiers; names training
  small models to use the RLM as the open problem.
- Alizadeh et al. *Recursive Language Models Meet Uncertainty: self-reflective program search*.
  arXiv:2603.15653. [Paper](https://arxiv.org/abs/2603.15653).
- Bertsch et al. *Oolong: Evaluating Long Context Reasoning and Aggregation*. arXiv:2511.02817.
  [Paper](https://arxiv.org/abs/2511.02817v1).

## Adaptive computation and routing collapse

- Kassenaar, Yang, François-Lavet. *Learning When to Think: Adaptive Reasoning for Test-Time
  Compute Allocation*. arXiv:2608.20256. [Paper](https://arxiv.org/html/2608.20256). Names routing
  collapse; balance term, per-mode caps, forced warmup.
- Fang, Ma, Wang. *Thinkless: LLM Learns When to Think*. arXiv:2505.13379.
  [Paper](https://arxiv.org/pdf/2505.13379). Decoupled GRPO for the control token.
- *Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL*
  (AutoThink). arXiv:2505.10832. [Paper](https://arxiv.org/pdf/2505.10832).
- Sun et al. *LLM Agents Already Know When to Call Tools*. arXiv:2605.09252.
  [Paper](https://arxiv.org/abs/2605.09252), [project](https://lilywenglab.github.io/when2tool/).

## RL algorithms and their biases

- Liu et al. *Understanding R1-Zero-Like Training* (Dr. GRPO). arXiv:2503.20783.
  [Paper](https://arxiv.org/abs/2503.20783).
- Ding et al. *On the Impossibility of Unbiased and Length-Invariant Policy Optimization with
  Outcome Rewards*. arXiv:2607.23364. [Paper](https://arxiv.org/abs/2607.23364).
- Guo et al. *Segment Policy Optimization*. arXiv:2505.23564. [Paper](https://arxiv.org/abs/2505.23564).
- Entropy and exploration collapse: STEER [arXiv:2510.10150](https://arxiv.org/abs/2510.10150),
  OPEFO [arXiv:2605.11491](https://arxiv.org/abs/2605.11491), Anchored Policy Optimization
  [arXiv:2602.05717](https://arxiv.org/pdf/2602.05717).
- Thinking Machines, *On-Policy Distillation* (October 2025):
  [collection](https://github.com/chrisliu298/awesome-on-policy-distillation).

## Harness-side learning

- Agrawal et al. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*.
  arXiv:2507.19457 (ICLR 2026). [Paper](https://arxiv.org/abs/2507.19457).
- *Meta-Harness: End-to-End Optimization of Model Harnesses*. arXiv:2603.28052.
  [Paper](https://arxiv.org/abs/2603.28052). *Harness-1*. arXiv:2606.02373.
  [Paper](https://arxiv.org/abs/2606.02373).
- *Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks*
  (NeurIPS 2025). [Paper](https://proceedings.nips.cc/paper_files/paper/2025/hash/5d1f02132ef51602adf07000ca5b6138-Abstract-Conference.html).
- SkillRL [arXiv:2602.08234](https://arxiv.org/pdf/2602.08234); AutoRefine
  [arXiv:2601.22758](https://www.alphaxiv.org/abs/2601.22758); survey of agent adaptation
  [arXiv:2512.16301](https://arxiv.org/pdf/2512.16301).

## Prior art added 22 October 2026

λ-RLM, a fixed combinator library for RLMs (https://arxiv.org/abs/2603.20105); Recursive Agent
Optimization (https://arxiv.org/abs/2605.06639); SRLM (https://arxiv.org/abs/2603.15653); PEEK
(https://arxiv.org/abs/2605.19932); Prime Agent (https://arxiv.org/abs/2608.23552); GRASP
(https://arxiv.org/abs/2605.29668); SkillGen (https://arxiv.org/abs/2605.10999); ARISE
(https://arxiv.org/abs/2606.02994); library-learning critiques (https://arxiv.org/abs/2410.20274,
https://arxiv.org/abs/2504.03048, https://arxiv.org/abs/2507.22069); ContinualSkillBench
(https://arxiv.org/abs/2608.03874); QwenLong-L1.5 (https://arxiv.org/abs/2512.12967); coding agents as
long-context processors (https://arxiv.org/abs/2603.20432); Recursive Agent Harnesses
(https://arxiv.org/abs/2606.13643); LCM (https://arxiv.org/abs/2605.04050).
