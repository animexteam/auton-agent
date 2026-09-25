"""agentcore — a lightweight but capable autonomous agent runtime.

Design goals (see ARCHITECTURE.md):
  * model-agnostic reasoning layer
  * a real agentic loop (understand -> plan -> act -> observe -> adapt -> verify)
  * a typed tool registry with validated contracts
  * progressive-disclosure skills instead of one giant system prompt
  * persistence that survives restarts even on an ephemeral filesystem
  * least-privilege execution with a controlled sandbox
"""

__version__ = "1.0.0"
