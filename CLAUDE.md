# gpu_llm_manager

One service for a multi-GPU server: GPU telemetry and reservations, HuggingFace model downloads, vLLM serving and
an agent gateway. Two front ends over the same core: web page (humans, phone) and MCP (agents).

## Phases
1. GPUs: telemetry, reservations, journal, history — done.
2. HuggingFace models: search, fit estimate, download queue, delete — done.
3. vLLM: a model per systemd --user unit `gm-model-<name>` (survives manager restarts); auto port = lowest free in
   `vllm.port_range`; several models per GPU; defaults = all free memory + model's max context; known start
   failures auto-retried; good params → `data/model_profiles.json` — done.
4. Agents: gateway `/v1` by model name, `model@prompt` = named system prompt, MCP `llm_ask` — done.

## Layout
- `gpu_manager/core.py` GPU facade · `models.py` + `hub.py` HF models · `runner.py` vLLM units · `gateway.py` +
  `prompts.py` agent access · `web*.py` page API · `mcp_*.py` MCP tools ·
  `messages.py` refusal codes (en + uk) · `static/index.html` the page · `agent_guide.md` text of MCP `gpu_guide`.
- `docs/specs/SPEC-GPU-00N-*.md` — behaviour + interface contract per phase.
- `deploy/gpu-manager.service` — systemd --user unit.

## Run
- `.venv/bin/python -m gpu_manager` (one instance: `data/manager.lock`); as a service: `systemctl --user restart gpu-manager`.
- `config.json` and `secrets.json` are git-ignored; start from `*.example.json`. Every setting is
  `{"value": …, "comment": "…"}`. Never read or print `secrets.json`.
- vLLM lives in a separate venv `.venv-vllm` (`requirements-vllm.txt`); it needs `vllm.cuda_home` with nvcc ≥ 12.8
  (FlashInfer JIT fails with an old system nvcc). Manual probe runs: only on the GPUs allowed for tests.

## Check
- `CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python -m pytest tests -q` (`GPU_MANAGER_LIVE=1` adds live NVML/HF tests).
- `.venv/bin/ruff check gpu_manager/` · `.venv/bin/pyright --pythonpath .venv/bin/python gpu_manager/`.
- Tests are written blind from `docs/specs/` by a separate agent. Do not fit tests to code: behaviour change →
  update the spec first, then the tests.

## Rules
- Web page and its error texts: Ukrainian. MCP, logs, exceptions: English. Code comments: Ukrainian.
- MCP results: one-line compact JSON (agent tokens). New refusal → code + both texts in `messages.py`.
- Bind only concrete addresses (`0.0.0.0` is refused): no auth by design, trusted users on a private network.
- Public repo: no IPs, host names, user names, tokens or home paths in tracked files.
- Minimum text everywhere: only what the next reader cannot do without.
