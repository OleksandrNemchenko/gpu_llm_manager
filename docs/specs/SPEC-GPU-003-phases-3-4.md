---
id: SPEC-GPU-003
status: draft
owner: gpu-manager
verified: 2026-09-23 22:50
---

# Фази 3–4: моделі на vLLM і доступ агентів

Спільне (конфіг, журнал, HTTP-захист, формат відмов, MCP) — SPEC-GPU-001/002.

## 1. Конфіг `vllm` (необов'язкові, крім `port_range`)
`bin` (`.venv-vllm/bin/vllm`, відносно теки конфігу), `cuda_home`, `default_fraction` (1.0; поза [0.05, 1] → `ConfigError`), `start_timeout_s` (900), `max_num_seqs` (32; < 1 → `ConfigError`).
Поля `Config`: `vllm_bin` (абсолютний), `vllm_cuda_home`, `vllm_default_fraction`, `vllm_start_timeout_s`, `vllm_max_num_seqs`.

## 2. `gpu_manager.runner.ModelRunner(cfg, store, journal, models, gpus, launcher, probe, port_free, clock)`
Шви: `launcher.start(unit, argv, env, log: Path)`, `.stop(unit) -> bool | None` (лише `False` — невдача),
`.active(unit) -> bool | None` (`None` — невідомо, прохід пропускається); `probe.health(port) -> bool`,
`.metrics(port) -> {running, waiting, kv_usage}`; `port_free(port) -> bool`. Юніт: `gm-model-<name>`.
`gpus` — `GpuManager`; `models` — `ModelStore`. `build_runner(cfg, manager, models)` підключає `models.in_use` і `manager.models_on`.

| # | Правило |
|---|---|
| 1 | `start(repo, gpu, user, fraction=None, max_model_len=None, extra_args=None, name=None, port=None)`; відмови: `unknown_user`, `unknown_gpu`, `reserved_by_other` (карта заброньована іншим), `model_not_local` (не `ready`), `bad_fraction` (поза (0,1]), `bad_extra_args` (не рядки або `--host --port --served-model-name --gpu-memory-utilization`), `bad_name`, `name_taken`, `no_free_port`. |
| 2 | Порт — явний (поза діапазоном → `bad_port`; зайнятий активною моделлю чи `port_free` = False → `port_taken`) або найменший у `vllm.port_range`, не зайнятий активною моделлю і з `port_free(port)`. `free_ports()` — усі такі порти. |
| 3 | Назва за замовчуванням — остання частина `repo` малими літерами; зайнята → `-2`, `-3`… |
| 4 | Вільна частка карти = донизу до сотих `min((total − used − 256 MiB)/total, 1 − Σ fraction активних моделей на цій карті, default_fraction)`. Частка: явна (з запиту) → інакше `min(профіль або default_fraction, вільна)`. `max_model_len` і `extra_args`: явні → профіль (`data/model_profiles.json`, ключ `repo`, читається при кожному старті) → не задається (максимум моделі) / `[]`. «Активні» = `starting` + `running`; при автоповторі власна частка моделі в суму не входить. |
| 5 | Явна частка більша за вільну → `gpu_memory_low`; неявна менша за 0.05 → `gpu_memory_low`. |
| 6 | argv: `<bin> serve <hf_home>/hub/models--<org>--<name>/snapshots/<rev> --host 127.0.0.1 --port P --served-model-name N --gpu-memory-utilization F --disable-uvicorn-access-log [--max-model-len L] [--max-num-seqs <vllm.max_num_seqs>, якщо в extra_args немає `--max-num-seqs` / `--max_num_seqs` у будь-якій формі, зокрема `=N`] [extra_args]`; env: `CUDA_VISIBLE_DEVICES=<gpu>`, `HF_HOME`, `HF_HUB_OFFLINE=1`, `CUDA_HOME`, `PATH=<cuda>/bin:…`. Лог: `data/servers/<name>.log`, кожен старт — рядок-маркер `=== gpu-manager start`. |
| 7 | Стани: `starting` → `running` (юніт активний і `health`) → зберігається профіль `{fraction, max_model_len, extra_args, gpu, updated, attempts}` (частка — лише виведена менеджером; явна з запиту лишає попередню частку профілю); юніт неактивний → автоповтор (п. 8) або `failed` з `error_code`/`error_params`; `starting` довше `start_timeout_s` → стоп і `failed` `hint_start_timeout`. Журнал: `model_start`, `model_stop`, `model_retry` (`change`), `model_failed` (`error`). |
| 8 | Автоповтор (до 5 спроб разом із першою). Рядок помилки — з `ERROR`, `…Error`, `…Exception`, `Traceback` або `fatal`. «estimated maximum model length is N» (у будь-якому рядку помилки) → `max_model_len` = N донизу до кратного 1024 (N < 1024 — N як є; N < 256 — як «No available memory»); «exceeds available Mamba cache blocks (N)» → `--max-num-seqs N` в `extra_args` (замінює попереднє в будь-якій формі; потрапляє в профіль разом з `extra_args`); далі за першим збігом у порядку правил: «less than desired GPU memory utilization» або OOM → частка на щабель нижче з (1.0, 0.95, 0.92, 0.9); якщо нижче нікуди, OOM / «KV cache» → `max_model_len` удвічі менший (≥ 2048; не заданий → 8192); «No available memory for the cache blocks» → частка ×2 у межах вільної; помилка компіляції (`Ninja build failed`, `nvcc fatal`, `InductorError`, `torch._dynamo`) → `--enforce-eager`. Одна спроба — одна зміна й один запис `auto_changes`. |
| 9 | `stop(name, user)`: активна → стоп, `stopped`, не піднімається після рестарту; `launcher.stop` дав `False` → `stop_failed`, модель лишається під наглядом (`starting`); `failed` → прибирається зі списку; інакше `server_not_found`. `active() → None` пропускає лише цю модель у цьому проході. |
| 10a | `move(name, user, gpu=None, port=None)`: неактивна → `server_not_found`; без змін → поточний стан; невідома карта → `unknown_gpu`, на новій карті бракує вільної частки (явна частка моделі або 0.05) → `gpu_memory_low` — обидва до зупинки; інакше `stop` → `start` з тією ж назвою, `max_model_len`, `extra_args` і явною часткою (виведену рахує заново); новий старт відмовив → старт на старому місці й та сама відмова. Журнал `model_move` (`gpu`, `port`, `from_gpu`, `from_port`). Помилки порту (`bad_port`, `port_taken`) — від нового старту, з поверненням. Власна частка при перенесенні на тій самій карті не рахується (модель уже зупинена). `unknown_user` — до зупинки. Успіх → стан нового запуску; не вдалося й повернення — модель зупинена, видно в журналі. `free_ports()` — за зростанням, HTTP — JSON-список. |
| 10 | `restore()` при старті менеджера: активні записи без живого юніта запускаються знову з тими ж параметрами й портом. |
| 11 | `servers()` — активні й `failed`, за (gpu, port); `on_gpu(gpu)` — `[{name, port, status, user}]`; `in_use(repo)`; `logs(name, lines, errors_only)` → `{name, status, hint, hint_params, lines}`. |
| 12 | `ModelStore.delete` запущеної моделі → `model_running`. Картка GPU (`/api/overview`) має поле `models` = `on_gpu`. |

Коди підказок: `hint_mamba_seqs` (`n`), `hint_kv_len` (`n`), `hint_kv_too_small`, `hint_no_kv_memory`, `hint_gpu_memory_taken`, `hint_oom`,
`hint_remote_code` («trust_remote_code=True»), `hint_port_in_use`, `hint_compile`, `hint_unsupported` («not supported»),
`hint_start_timeout`, `hint_see_logs`.

## 3. Доступ агентів
- `gpu_manager.prompts.PromptStore(users, store, journal, clock)`: `save(name, text, user)` (назва як у п. 2.1,
  текст непорожній ≤ 20000 → інакше `bad_prompt`), `delete(name, user)`, `get(name)` (`prompt_not_found`),
  `list()` → `[{name, preview (80 символів), user, updated}]`. Журнал `prompt_save`, `prompt_delete`.
- `gpu_manager.gateway`: `resolve(runner, prompts, body) -> (port, body)`: `model` = `назва[@промпт]`, лише `running`,
  інакше `model_not_running`; промпт — першим повідомленням `system` (для `prompt` — префікс тексту).
  Маршрути: `GET /v1/models` (`{object: list, data: [{id, object: model, owned_by}]}`), `POST /v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings` — пересилання на `127.0.0.1:<port>` (потік як є); помилка —
  `{error: {message, type, code}}`, 404 для `model_not_running`/`prompt_not_found`, 400 інше.

## 4. HTTP і MCP
- HTTP: `GET /api/servers`, `POST /api/servers/start` `{repo, gpu, user, fraction?, max_model_len?, extra_args?, name?, port?}`,
  `POST /api/servers/move` `{name, user, gpu?, port?}`, `GET /api/servers/ports` (→ `free_ports()`),
  `POST /api/servers/stop` `{name, user}`, `GET /api/servers/logs?name=&lines=&errors=1`; поле `hint` — текст українською.
- MCP (є лише з runner / prompts): `model_start` (+`port`), `model_move`, `model_stop`, `models_running`, `model_logs`, `prompt_save`,
  `prompts_list`, `prompt_delete`, `llm_ask(model, prompt, system_prompt?, system?, max_tokens=1024, temperature=0.2)`
  → лише текст відповіді. Підказка в MCP — `"<code>: <English>"`.
- `build_app(cfg, manager, models, runner, prompts)`, `build_mcp(..., runner=, prompts=)`.
