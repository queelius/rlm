// Fresh SFT start. BOS-only behavior is provided by launch_bos_only.py.
// EOS remains absent despite the historical flag; this is a separate open issue.
local historical = import '/project/alex_phd/runs/spo-reproduction-20261008/author-training-config.json';
local sft = '/project/alex_phd/research-cache/spo/models/models--realtreetune--rho-1b-sft-GSM8K/snapshots/b28fda3216f8178c1f37c3c764196e3ca2439a30';
local dataset = '/project/alex_phd/research-cache/spo/assets/gsm8k-author-20241002/gsm8k';
local stage = std.extVar('APP_DIRECTORY');
local resolve_paths(value) =
  if std.type(value) == 'object' then
    { [key]: resolve_paths(value[key]) for key in std.objectFields(value) }
  else if std.type(value) == 'array' then
    [resolve_paths(item) for item in value]
  else if std.type(value) == 'string' && value == 'realtreetune/rho-1b-sft-GSM8K' then sft
  else if std.type(value) == 'string' && value == 'data/gsm8k' then dataset
  else value;
resolve_paths(historical) + {
  num_iterations: 1000,  // Preserve 16000-update / 480-warmup LR horizon.
  early_stop_iteration: 690,
  directory: stage,
  exp_name: 'native',
  global_vars+: {
    seed: 42,
    dirs+: {experiments: stage, data: '/project/alex_phd/research-cache/spo/assets/gsm8k-author-20241002'},
  },
  episode_generator+: {vllm_gpu_memory_utilization: 0.3},
}
