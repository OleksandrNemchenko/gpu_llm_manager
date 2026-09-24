# gpu-manager: agent guide

GPU server shared by {users}. You act for ONE human; `user` in every tool = their Ubuntu login.
If you do not know it, ask the human once and keep it (e.g. a line in CLAUDE.md / AGENTS.md).

## Cards
- Which are free: `gpu_free`. Who uses a card: `gpu_who`. Everything: `gpu_status`.
- Before long GPU work: `gpu_reserve(gpu, user, purpose, hours)`. When done: `gpu_release(gpu, user)`.
- Treat `busy` (processes without a reservation) and `reserved_expired` as taken; tell the human.
- Never release someone else's reservation (`force=true`) without the human's explicit OK.

## Files
Inbox on the server: `{inbox}/<user>/`. {inbox_status}
- Write, move and delete ONLY inside your human's folder; never touch other users' folders.
- One subfolder per task; `rm -r` it when the task is done.
- Copy on the same server:
  `rsync -a --chmod=Dg+rx,Fg+r <src>/ {inbox}/<user>/<task>/`
- Copy from another machine (SSH over Tailscale):
  `rsync -a --partial --chmod=Dg+rx,Fg+r <src>/ <user>@{host}:{inbox}/<user>/<task>/`
- `--chmod` is required: without group read the server cannot read private files.
- Never paste file contents into tool arguments: pass paths, the server reads files itself.

## Models
- Find: `hf_search`, check fit: `hf_model`, download: `model_download` (poll `model_downloads`).
- Start: `model_start(repo, gpu, user)`; poll `models_running` until `running`. By default the model takes ALL free
  memory of the GPU and its maximum context; the server itself retries known failures (see `auto_changes`).
  On `failed`: read the hint and `model_logs`, adjust `fraction` / `max_model_len` / `extra_args`, start again.
  A good start is remembered in the model's profile.
- To share a GPU between models, give each an explicit `fraction` (e.g. 0.45). Stop with `model_stop`.

## Using models
- Quick question from you: `llm_ask(model, prompt, system_prompt=<saved name>)` — returns only the text.
- Programs and other agents: OpenAI-compatible `http://{host}:{port}/v1` (`/v1/models`, `/v1/chat/completions`),
  model = its name from `models_running`; `<model>@<prompt name>` adds a saved system prompt.
- Save a long system prompt once with `prompt_save`, then refer to it by name (saves your tokens).

## Economy
Answers are compact JSON. Ask for one card (`gpu_who`) rather than all; keep `gpu_history` `points` small.
