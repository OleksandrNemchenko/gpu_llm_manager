# gpu-manager: agent guide

GPU server shared by {users}. You act for ONE human; `user` in every tool = their Ubuntu login.
If you do not know it, ask the human once and keep it (e.g. a line in CLAUDE.md / AGENTS.md).

## Cards
- Which are free: `gpu_free`. Who uses a card: `gpu_who`. Everything: `gpu_status`.
- Before long GPU work: `gpu_reserve(gpu, user, purpose, hours)`. When done: `gpu_release(gpu, user)`.
- Treat `busy` (processes without a reservation) and `reserved_expired` as taken; tell the human.
- Never release someone else's reservation (`force=true`) without the human's explicit OK.

## Files for models
Your folder: `{inbox}/<user>/` (`files_inbox(user)` returns it with the copy commands). {inbox_status}
- Write, move and delete ONLY inside your human's folder; one subfolder per task; `rm -r` it when the task is done.
- Copy on the same server: `rsync -a --chmod=Dg+rx,Fg+r <src> {inbox}/<user>/<task>/`
- Copy from another machine (SSH): `rsync -a --partial --chmod=Dg+rx,Fg+r <src> <user>@{host}:{inbox}/<user>/<task>/`
- `--chmod` is required: the server and the models read your files through the group.
- Then pass only paths: `llm_ask(model, prompt, files=[...])` — images, audio and video are read by the model itself,
  text and PDF are inserted by the server. Never paste file contents yourself: that costs your tokens.
- `/v1` clients: `image_url` / `audio_url` / `video_url` with `file://{inbox}/<user>/<task>/<file>`.

## Models
- Find: `hf_search`, check fit: `hf_model`, download: `model_download` (poll `model_downloads`).
- Start: `model_start(repo, gpu, user)`; poll `models_running` until `running`. By default the model takes ALL free
  memory of the GPU and its maximum context; the server itself retries known failures (see `auto_changes`).
  On `failed`: read the hint and `model_logs`, adjust `fraction` / `max_model_len` / `extra_args`, start again.
  A good start is remembered in the model's profile.
- To share a GPU between models, give each an explicit `fraction` (e.g. 0.45). Stop with `model_stop`.
- For agentic clients (OpenCode and the like) the model must call tools: start it with `extra_args=["--enable-auto-tool-choice",
  "--tool-call-parser", "<parser>"]` (parser from the model card: `qwen3_coder` for recent Qwen, `hermes` for Qwen2.5,
  `llama3_json` for Llama 3, `openai` for gpt-oss). Without it every request with tools fails.
- Move a running model to another GPU or port: `model_move(name, user, gpu, port)` — a restart under the same name;
  if the new start fails, the model returns to its old place.

## Using models
- Quick question from you: `llm_ask(model, prompt, system_prompt=<saved name>)` — returns only the text.
- Programs and other agents: OpenAI-compatible `http://{host}:{port}/v1` (`/v1/models`, `/v1/chat/completions`),
  model = its name from `models_running`; `<model>@<prompt name>` adds a saved system prompt.
- Save a long system prompt once with `prompt_save`, then refer to it by name (saves your tokens).

## Economy
Answers are compact JSON. Ask for one card (`gpu_who`) rather than all; keep `gpu_history` `points` small.
