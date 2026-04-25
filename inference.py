"""
Deterministic baseline inference script for the DryLabSim RL environment.
=========================================================================

This baseline uses the OpenAI client for action selection at each step. The
environment computes the canonical final score server-side via
grade_episode(obs, latent) and exposes it in terminal observation metadata.

MANDATORY env vars typically used by the hackathon harness:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Environment-specific extra variable for this HTTP-served RL environment:
    ENV_URL        The base URL of the deployed environment server.

STDOUT FORMAT (mandatory — automated grader parses these lines):
    [START] task=<task_name> env=drylabsim model=<model_name>
    [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""
