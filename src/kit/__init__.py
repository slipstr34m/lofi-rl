"""kit — reusable pieces for laptop-driven projects on USC CARC Discovery.

  kit.vllm_client  OpenAI-compatible client for local vLLM servers (API key, retries, round-robin)
  kit.hf_cache     resolve Hugging Face repo ids to local snapshot dirs (offline jobs)
  kit.bench        strong-scaling benchmark harness for embarrassingly parallel work
  kit.bench_vllm   GPU-throughput benchmark: calls/min against one or more vLLM servers
  kit.chain        chained long runs: SLURM deadline, complete-unit checkpoints, atomic latest, DONE/STOP
  kit.gates        strict human-gate approval check (a dated line of its own, never the instructions)
  kit.publish      Hugging Face dataset publishing: dry run by default, --go + approved gate to upload
"""

__version__ = "1.0.0"
