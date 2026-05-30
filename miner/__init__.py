"""PRL-3090 miner orchestration (cold path).

This package is the Python orchestration layer described in PRD §11.1/§16/§17. It
deliberately mirrors the official Pearl architecture: a Python cold path (job fetch,
submit, metrics, safety — none of it performance-critical) driving a C++/CUDA hot
path (the GPU NoisyGEMM kernels). The "no Python in the hot loop" rule (PRD §18) is
satisfied because the hot loop lives in the CUDA backend, not here.

Conceptual mapping to the PRD §20 module names:
  protocol/gateway_client  -> node_client + stratum_client (gateway speaks for the node)
  runtime.JobManager       -> job_manager
  runtime.Submitter        -> submitter
  runtime.Metrics          -> metrics
  runtime.SafetyMonitor    -> safety
  config                   -> config
"""

__version__ = "0.1.0"
