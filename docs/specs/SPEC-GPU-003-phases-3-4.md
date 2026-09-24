---
id: SPEC-GPU-003
status: accepted
owner: gpu-manager
verified: 2026-09-24 12:06
---

# Фази 3–4: моделі на vLLM і доступ агентів

Спільне (конфіг, журнал, HTTP-захист, формат відмов, MCP) — SPEC-GPU-001/002.

## 1. Конфіг `vllm` (необов'язкові, крім `port_range`)
`bin` (`.venv-vllm/bin/vllm`, відносно теки конфігу), `cuda_home`, `default_fraction` (0.95 — запас на обчислення під час відповіді; поза [0.05, 1] → `ConfigError`), `start_timeout_s` (900; ≤ 0 → `ConfigError`), `max_num_seqs` (32; < 1 → `ConfigError`).
Поля `Config`: `vllm_bin` (абсолютний), `vllm_cuda_home`, `vllm_default_fraction`, `vllm_start_timeout_s`, `vllm_max_num_seqs`.

## 2. `gpu_manager.runner.ModelRunner(cfg, store, journal, models, gpus, launcher, probe, port_free, clock)`
Шви: `launcher.start(unit, argv, env, log: Path)`, `.stop(unit) -> bool | None` (лише `False` — невдача),
`.active(unit) -> bool | None` (`None` — невідомо, прохід пропускається); `probe.health(port) -> bool`,
`.metrics(port) -> {running, waiting, kv_usage}`; `port_free(port) -> bool`. Юніт: `gm-model-<name>.service`
(суфікс явний: назви `x` і `x.service` — різні юніти).
`gpus` — `GpuManager`; `models` — `ModelStore`. `build_runner(cfg, manager, models)` підключає `models.in_use` і `manager.models_on`.

| # | Правило |
|---|---|
| 1 | `start(repo, gpu, user, fraction=None, max_model_len=None, extra_args=None, name=None, port=None)`; відмови: `unknown_user`, `unknown_gpu`, `reserved_by_other` (карта заброньована іншим), `model_not_local` (не `ready`), `bad_fraction` (поза (0,1]), `bad_extra_args` (не список рядків або прапорець із `--host --port --uds --served-model-name --gpu-memory-utilization --kv-cache-memory-bytes --num-gpu-blocks-override --config --allowed-local-media-path --allowed-media-domains --api-key --root-path --ssl-keyfile --ssl-certfile --ssl-ca-certs --middleware` — також скорочений, з `_` замість `-`, у формі `=значення` чи `--прапорець.ключ`), `bad_name`, `name_taken`, `no_free_port`. |
| 2 | Порт — явний (поза діапазоном → `bad_port`; зайнятий активною моделлю чи `port_free` = False → `port_taken`) або найменший у `vllm.port_range`, не зайнятий активною моделлю і з `port_free(port)`. `free_ports()` — усі такі порти. |
| 3 | Правило назви (моделі й промпту): `[a-z0-9][a-z0-9._-]{0,47}`, інакше `bad_name`. Назва за замовчуванням — остання частина `repo` малими літерами, недозволені символи → `-`, не довше 40, без `-`, `.`, `_` на краях (порожня → `model`); зайнята → `-2`, `-3`… |
| 4 | Вільна частка карти = донизу до сотих `min((total − used − 256 MiB)/total, 1 − Σ fraction активних моделей на цій карті)`; для неявної частки — ще й не більше `default_fraction` (стеля лише для частки, яку вибирає менеджер). Частка: явна (з запиту) → інакше `min(профіль або default_fraction, вільна)`. `max_model_len` і `extra_args`: явні → профіль (`data/model_profiles.json`, ключ `repo`, читається при кожному старті) → не задається (максимум моделі) / `[]`. «Активні» = `starting` + `running`; при автоповторі власна частка моделі в суму не входить. |
| 5 | Явна частка більша за вільну → `gpu_memory_low`; неявна менша за 0.05 → `gpu_memory_low`. |
| 6 | argv: `<bin> serve <hf_home>/hub/models--<org>--<name>/snapshots/<rev> --host 127.0.0.1 --port P --served-model-name N --gpu-memory-utilization F --disable-uvicorn-access-log [--allowed-local-media-path <files.inbox_dir>, якщо тека існує] [--max-model-len L] [--max-num-seqs <vllm.max_num_seqs>, якщо в extra_args немає `--max-num-seqs` / `--max_num_seqs` у будь-якій формі, зокрема `=N`] [extra_args]`; env: `CUDA_DEVICE_ORDER=PCI_BUS_ID` (номер карти в CUDA = номер у NVML), `CUDA_VISIBLE_DEVICES=<uuid карти з NVML>` (номер карти — якщо `uuid` порожній; CUDA не рахує відпалої карти, і номери решти зсунулися б), `HF_HOME`, `HF_HUB_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `CUDA_HOME`, `PATH=<cuda>/bin:…`. Лог: `data/servers/<name>.log`, кожен старт — рядок-маркер `=== gpu-manager start`. Справжній `SystemdLauncher` запускає команду через `sg <група теки> -c "exec …"`, коли група теки файлів — не основна група користувача сервісу і він у ній: процес моделі читає теки користувачів (права 2750). |
| 7 | Стани: `starting` → `running` (юніт активний і `health`) → зберігається профіль `{fraction, max_model_len, extra_args, gpu, updated, attempts}` (частка — лише виведена менеджером; явна з запиту лишає попередню частку профілю); юніт неактивний → автоповтор (п. 8) або `failed` з `error_code`/`error_params`; `starting` довше `start_timeout_s` → стоп і `failed` `hint_start_timeout`. Журнал: `model_start`, `model_stop`, `model_retry` (`change`), `model_failed` (`error`). |
| 8 | Автоповтор (до 5 спроб разом із першою). Рядок помилки — з `ERROR`, `…Error`, `…Exception`, `Traceback` або `fatal`. «estimated maximum model length is N» (у будь-якому рядку помилки) → `max_model_len` = N донизу до кратного 1024 (N < 1024 — N як є; N < 256 — як «No available memory»); «greater than the derived max_model_len (<ключ>=N» у будь-якому рядку логу останнього старту (vLLM друкує її в рядку pydantic `  Value error, User-specified max_model_len …` без слова Error) → `max_model_len` = N (`hint_ctx_over_model`, `n`: заданий контекст більший за межу моделі); «exceeds available Mamba cache blocks (N)» → `--max-num-seqs N` в `extra_args` (замінює попереднє в будь-якій формі; потрапляє в профіль разом з `extra_args`); далі за першим збігом у порядку правил: «less than desired GPU memory utilization» або OOM → частка на щабель нижче з (1.0, 0.95, 0.92, 0.9); якщо нижче нікуди, OOM / «KV cache» → `max_model_len` удвічі менший (≥ 2048; не заданий → 8192); «No available memory for the cache blocks» → частка ×2 у межах вільної; помилка компіляції (`Ninja build failed`, `nvcc fatal`, `InductorError`, `torch._dynamo`) → `--enforce-eager`. Одна спроба — одна зміна й один запис `auto_changes`. |
| 9 | `stop(name, user)`: активна → стоп, `stopped`, не піднімається після рестарту; `launcher.stop` дав `False` → `stop_failed`, модель лишається під наглядом (`starting`); `failed` → прибирається зі списку; інакше `server_not_found`. `active() → None` пропускає лише цю модель у цьому проході. `stop`, що прийшов, поки юніт ще запускається (після запису `starting`, до кінця `launcher.start` — і при першому старті, і при автоповторі), не лишає живого юніта: менеджер зупиняє його одразу після `launcher.start`. |
| 10a | `move(name, user, gpu=None, port=None)`: неактивна → `server_not_found`; без змін → поточний стан; невідома карта → `unknown_gpu`, на новій карті бракує вільної частки (явна частка моделі або 0.05) → `gpu_memory_low` — обидва до зупинки; інакше `stop` → `start` з тією ж назвою, `max_model_len`, `extra_args` і явною часткою (виведену рахує заново); новий старт відмовив → старт на старому місці й та сама відмова. Журнал `model_move` (`gpu`, `port`, `from_gpu`, `from_port`). Помилки порту (`bad_port`, `port_taken`) — від нового старту, з поверненням. Власна частка при перенесенні на тій самій карті не рахується (модель уже зупинена): після зупинки менеджер бере свіжий замір карт (`GpuManager.tick()`), і новий старт та повернення рахують вільну пам'ять з нього, а не з заміру, коли модель ще працювала. `unknown_user` — до зупинки. Успіх → стан нового запуску; не вдалося й повернення — модель зупинена, видно в журналі. `free_ports()` — за зростанням, HTTP — JSON-список. |
| 10 | `restore()` при старті менеджера: активні записи без живого юніта запускаються знову з тими ж параметрами й портом. |
| 11 | `servers()` — активні й `failed`, за (gpu, port); `on_gpu(gpu)` — `[{name, port, status, user}]`; `in_use(repo)`; `logs(name, lines, errors_only)` → `{name, status, hint, hint_params, lines}`. |
| 12 | `ModelStore.delete` запущеної моделі → `model_running`. Картка GPU (`/api/overview`) має поле `models` = `on_gpu`. |

Коди підказок: `hint_mamba_seqs` (`n`), `hint_kv_len` (`n`), `hint_ctx_over_model` (`n`), `hint_kv_too_small`, `hint_no_kv_memory`, `hint_gpu_memory_taken`, `hint_oom`,
`hint_remote_code` («trust_remote_code=True»), `hint_port_in_use`, `hint_compile`, `hint_unsupported` («not supported»),
`hint_start_timeout`, `hint_see_logs`.

## 3. Доступ агентів
- `gpu_manager.prompts.PromptStore(users, store, journal, clock)`: `save(name, text, user)` (назва як у п. 2.3,
  текст непорожній ≤ 20000 → інакше `bad_prompt`), `delete(name, user)`, `get(name)` (`prompt_not_found`),
  `list()` → `[{name, preview (80 символів), user, updated}]`. Журнал `prompt_save`, `prompt_delete`.
- `gpu_manager.gateway`: `resolve(runner, prompts, body) -> (port, body)`: `model` = `назва[@промпт]`, лише `running`,
  інакше `model_not_running`; промпт — першим повідомленням `system` (для `prompt` — префікс тексту).
  Маршрути: `GET /v1/models` (`{object: list, data: [{id, object: model, owned_by}]}`), `POST /v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings` — пересилання на `127.0.0.1:<port>` (потік як є); помилка —
  `{error: {message, type, code}}`, 404 для `model_not_running`/`prompt_not_found`, 400 інше. Клієнт, що розірвав
  з'єднання, поки модель ще не почала відповідати (зокрема без `stream`), обриває й запит до моделі: генерація
  не триває даремно.

## 4. HTTP і MCP
- HTTP: `GET /api/servers`, `POST /api/servers/start` `{repo, gpu, user, fraction?, max_model_len?, extra_args?, name?, port?}`,
  `POST /api/servers/move` `{name, user, gpu?, port?}`, `GET /api/servers/ports` (→ `free_ports()`),
  `POST /api/servers/stop` `{name, user}`, `GET /api/servers/logs?name=&lines=&errors=1`; поле `hint` — текст українською.
- MCP (є лише з runner / prompts): `model_start` (+`port`), `model_move`, `model_stop`, `models_running`, `model_logs`, `prompt_save`,
  `prompts_list`, `prompt_delete`, `llm_ask(model, prompt, system_prompt?, system?, max_tokens=1024, temperature=0.2)`
  → лише текст відповіді. Підказка в MCP — `"<code>: <English>"`.
- `build_app(cfg, manager, models, runner, prompts)`, `build_mcp(..., runner=, prompts=)`.
- `llm_ask`: модель не дала тексту → відмова `empty_answer` (`reason` = `finish_reason`). Файли — §6.

## 5. Чат на сторінці
- Вкладка «Чат»: розмова з будь-якою `running` моделлю через `/v1/chat/completions` (потік). Розмова живе в
  сховищі сесії браузера як дані (не HTML); модель, якої вже немає серед запущених, замінюється першою запущеною.
- Поки модель відповідає, нова репліка не надсилається; «Почати розмову» обриває відповідь, і її залишок у нову
  розмову не потрапляє.
- `POST /api/convert/pdf` `{data: data URL або base64, mode: "text" | "images"}` → `{pages, text}` (сторінки з
  позначкою `[page N]`) або `{pages, images: [data URL JPEG]}`. Відмова `bad_pdf` (`detail`): не base64, більше
  100 MiB, не PDF, сторінка не читається, сторінок понад 500 (`text`) чи 100 (`images`). Сторінка-картинка —
  не більше 2000 px по довшій стороні. Кілька конвертацій одночасно виконуються по черзі.

## 6. Файли для моделей (тека `files.inbox_dir`)
Агент кладе файл у свою теку й передає лише шлях: вміст іде в модель повз його токени.
- `gpu_manager.inbox.Inbox(root: Path, users, host)`. `info(user)` → `{dir, ready, file_url_prefix, copy_local,
  copy_remote, use}`: `dir` = `<root>/<user>/`, `ready` — ця тека існує, `file_url_prefix` = `file://<root>/<user>/`,
  `copy_*` — команди `rsync` (локально й `<user>@<host>:` по SSH); `unknown_user`.
- `parts(paths, pdf_mode="text")` → частини повідомлення OpenAI у порядку `paths` (шляхи абсолютні; будь-який файл
  теки, зокрема чужої — користувачі довірені; регістр розширення не важить; `<назва>` — ім'я файлу; хибний
  `pdf_mode` відмовляє завжди):
  - шлях, що після розкриття (зокрема символьних посилань) не лежить усередині `<root>/` або не є файлом → `bad_file_path` (`path`);
  - `.png .jpg .jpeg .webp .gif .bmp` → `{"type": "image_url", "image_url": {"url": "file://<розкритий шлях>"}}`;
    `.wav .mp3 .flac .ogg .m4a` → `audio_url`; `.mp4 .webm .mov .mkv .avi` → `video_url` (так само) — їх читає модель;
  - `.pdf`: `pdf_mode` `text` → `{"type": "text", "text": "[file <назва>, <N> pages]\n<текст>"}`; `images` → `image_url`
    з data URL JPEG на кожну сторінку; межі й відмови — §5 (`bad_pdf`); інший `pdf_mode` → `bad_request`;
  - решта — текст UTF-8 до 1 MiB → `{"type": "text", "text": "[file <назва>]\n<вміст>"}`; не UTF-8 чи більше → `bad_file` (`path`, `detail`);
  - понад 20 шляхів → `bad_file`.
- MCP `files_inbox(user)` → `info`. `llm_ask(..., files=[], pdf_mode="text")`: з `files` вміст повідомлення
  користувача — `parts(files)` і в кінці `{"type": "text", "text": prompt}`; без `files` — рядок `prompt`, як раніше.
- `build_mcp(..., inbox=None)`: `files_inbox` є лише з `inbox`; `files` без `inbox` → `bad_request`.
  `build_app` передає `Inbox(cfg.inbox_dir, cfg.users, <public_host без порту>)`.
- `/v1`: посилання `file://<root>/…` в `image_url` / `audio_url` / `video_url` модель читає сама.
