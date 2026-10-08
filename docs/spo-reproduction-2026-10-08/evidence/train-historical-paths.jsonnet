// Original Figure 3 training values recovered from author W&B run rbtvnfqm.
// Only immutable asset paths and experiment/runtime directory are adapted.
// Figure 3 comparison is prospectively fixed at completed checkpoint 690.
// num_iterations MUST remain 1000: it also controls the learning-rate schedule.
local historical = import 'author-training-config.json';
local sft = '/project/alex_phd/research-cache/spo/models/models--realtreetune--rho-1b-sft-GSM8K/snapshots/b28fda3216f8178c1f37c3c764196e3ca2439a30';
local dataset = '/project/alex_phd/research-cache/spo/assets/gsm8k-author-20241002/gsm8k';
local stage = '/project/alex_phd/runs/spo-reproduction-20261008/spo-historical-attempt1';

local resolve_paths(value) =
  if std.type(value) == 'object' then
    { [key]: resolve_paths(value[key]) for key in std.objectFields(value) }
  else if std.type(value) == 'array' then
    [resolve_paths(item) for item in value]
  else if std.type(value) == 'string' && value == 'realtreetune/rho-1b-sft-GSM8K' then
    sft
  else if std.type(value) == 'string' && value == 'data/gsm8k' then
    dataset
  else value;

resolve_paths(historical) + {
  // Fixed comparison endpoint; learning-rate horizon remains the original 1000.
  early_stop_iteration: 690,
  directory: stage,
  exp_name: 'native',
  global_vars+: {
    dirs+: {
      experiments: stage,
      data: '/project/alex_phd/research-cache/spo/assets/gsm8k-author-20241002',
    },
  },
}
