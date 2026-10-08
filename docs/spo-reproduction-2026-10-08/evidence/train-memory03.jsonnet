// Attempt1 exhausted memory while actor logprobs ran alongside vLLM's auto65% cache.
// Resource-only change; preserve all model, data, objective and sampling settings.
{
  directory: '/project/alex_phd/runs/spo-reproduction-20261008/spo-historical-memory03-attempt2',
  global_vars+: {dirs+: {experiments: '/project/alex_phd/runs/spo-reproduction-20261008/spo-historical-memory03-attempt2'}},
  episode_generator+: {vllm_gpu_memory_utilization: 0.3},
}
