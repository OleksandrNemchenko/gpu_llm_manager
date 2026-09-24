---
id: SPEC-GPU-002
status: accepted
owner: gpu-manager
verified: 2026-09-24 12:06
---

# Специфікація фази 2: моделі HuggingFace

Джерело — обговорення з користувачем 2026-09-23. Контракт поведінки й інтерфейсів; реалізація — `gpu_manager/`.
Тести пишуться лише з цього документа. Спільне з фазою 1 (конфіг, журнал, захист HTTP, формат відмов, мови) —
у `SPEC-GPU-001-phase-1.md`; тут лише нове.

## 1. Призначення

Знайти модель на HuggingFace (HF), оцінити до завантаження, чи влізе вона в карту сервера, завантажити її в кеш HF на
сервері у фоні (черга, прогрес, скасування, продовження після обриву чи перезапуску), показати завантажені й
видалити непотрібні. Оцінка «чи влізе» — умовна: остаточно покаже перший запуск (фаза 3).

## 2. Конфіг: розділ `models` (необов'язковий, формат налаштувань — як у SPEC-GPU-001 §2)

| # | Налаштування | Типово | Правило |
|---|---|---|---|
| 1 | `models.hf_home` | `<домашня тека>/hf-cache` | кеш HF (`HF_HOME`); моделі лежать у `<hf_home>/hub`; відносний — від теки конфігу, у `Config` — абсолютний |
| 2 | `models.max_parallel_downloads` | 2 | < 1 → `ConfigError` |
| 3 | `models.min_free_disk_gib` | 50 | запас диска, що має лишитися після завантаження; < 0 → `ConfigError` |
| 4 | `models.fit_memory_fraction` | 0.9 | поза (0, 1] → `ConfigError` |
| 5 | `models.fit_overhead_gib` | 2.0 | запас на активації в оцінці; < 0 → `ConfigError` |

Поля `Config`: `hf_home` (`Path`), `max_parallel_downloads`, `min_free_disk_gib`, `fit_memory_fraction`,
`fit_overhead_gib`, `secrets_path` (= `<тека конфігу>/secrets.json`).

Токен HF — `gpu_manager.credentials.hf_token(path) -> str | None`: значення `hf_token.value` з `secrets.json`;
`None`, якщо файлу немає, він не JSON, поля немає, значення порожнє або починається з `PASTE_` (заглушка).

## 3. Вибір файлів і оцінка (`gpu_manager.hub`, чисті функції)

| # | Функція | Правило |
|---|---|---|
| 1 | `select_files(names) -> list[str]` | відкидає `*.gguf *.pth *.pt *.onnx *.onnx_data *.h5 *.msgpack *.ot *.tflite *.mlmodel` і теки `original/ onnx/ openvino/ coreml/ gguf/`; якщо є хоч один `*.safetensors` — відкидає й `*.bin`; решту лишає; результат відсортований |
| 2 | `weight_bytes(files: dict[str,int]) -> int` | сума розмірів `*.safetensors` і `*.bin` |
| 3 | `kv_bytes_per_token(config) -> int \| None` | `2 × num_hidden_layers × num_key_value_heads × head_dim × 2`; `num_key_value_heads` за відсутності = `num_attention_heads`; `head_dim` за відсутності = `hidden_size // num_attention_heads`; якщо є `text_config` — рахується з нього; бракує полів → `None` |
| 4 | `estimate_fit(files, config, card_mib, fraction, overhead_gib) -> Fit` | для кожного різного обсягу карти `mib`: `usable = mib·2²⁰·fraction − weight_bytes − overhead_gib·2³⁰`; `fits = usable > 0`; `max_context_tokens = usable // kv` (обмежено `max_position_embeddings`, якщо він є), `0`, якщо `usable ≤ 0`, `None`, якщо `kv` невідомий |

`Fit`: `weights_gib` (2 знаки), `kv_kib_per_token` (1 знак або `None`), `max_context` (`max_position_embeddings`
або `None`), `per_card` (`{mib: {usable_gib (1 знак), fits, max_context_tokens}}`), `warnings` (англійською):
`quantization_config.quant_method`, що містить `fp4` (зокрема `nvfp4`, `mxfp4`) → попередження «no hardware support
below sm_100»; рівно `fp8` → попередження «weight-only W8A16» (тексти не називають модель карти); `kv` невідомий → попередження про контекст.
Серед файлів немає ваг (`weight_bytes = 0`, напр. репозиторій лише з GGUF) → `fits: false` для кожної карти й
попередження «no safetensors/bin weights».

## 4. Клієнт HF (`gpu_manager.hub.HubClient(token)`)

`token` — функція без аргументів, що повертає токен або `None` (читається щоразу). Методи:
`search(query, limit) -> list[{repo, downloads, likes, gated, task, params, updated}]` (за популярністю);
`files(repo, revision) -> (sha, {файл: розмір} після select_files, gated)`; `check_access(repo)`;
`config(repo, revision) -> dict` (config.json **без запису в кеш HF**; немає файла → `{}`).
Помилки HF → `ManagerError`: немає моделі чи ревізії — `hf_not_found`; немає доступу до gated-моделі — `hf_gated`;
мережа чи інша помилка HF — `hf_unavailable`. Кожен запит до HF обмежений тайм-аутом 20 с (`hf_unavailable`).

## 5. Сховище моделей (`gpu_manager.models.ModelStore`)

Конструктор: `ModelStore(cfg, hub, store, journal, card_mib, token, clock=time.time, spawn=subprocess.Popen)`;
`hub` — об'єкт з методами §4; `card_mib()` — список обсягів карт у MiB; `spawn` — як `subprocess.Popen`.

### 5.1 Стани завантаження

`queued` → `downloading` → `done` | `failed`; з `queued`/`downloading` — `cancelled`. Активні: `queued`, `downloading`.

| # | Правило |
|---|---|
| 1 | `download(repo, user, revision=None)`: `user` з `users.allowed`, інакше `unknown_user`. |
| 2 | Якщо та сама модель уже активна — новий запис не створюється, повертається поточний стан. |
| 3 | Якщо всі вибрані файли вже є в знімку ревізії — одразу стан `done`, процес не запускається, у журнал **не** пишеться. |
| 4 | Gated-модель без доступу → `hf_gated` (перевірка до постановки в чергу). |
| 4a | Серед вибраних файлів немає ваг (`weight_bytes = 0`) → `no_vllm_weights` (одразу після опису моделі з HF, раніше за пп. 3–5). |
| 5 | Потрібний обсяг = сума файлів − уже наявне; якщо `вільно − потрібно − залишок інших активних завантажень < min_free_disk_gib` → `disk_full`. Перед запуском процесу з черги — та сама перевірка, але із залишком лише тих, що вже качаються (`downloading`); не проходить → `failed` з `error_code` `disk_full`. |
| 6 | Інакше — запис `queued`, журнал `download` (`repo`, `size_gib`), одразу крок черги (`poll`). |
| 7 | `poll()`: процес, що завершився з кодом 0 → `done` (журнал `download_done`); інакше → `failed` з `error_code` і `error` (журнал `download_failed`, поле `error`); далі в порядку `started` запускаються `queued`, доки активних процесів < `max_parallel_downloads`. |
| 8 | `cancel(repo, user)`: лише для активного, інакше `download_not_active`; процес зупиняється (SIGTERM, через 5 с — SIGKILL); стан `cancelled`; журнал `download_cancel`. Недокачані файли лишаються. |
| 9 | Стан черги зберігається в `state.json`, розділ `downloads`. Після перезапуску `downloading` стає `queued` і продовжується. |
| 10 | `stop_all()` зупиняє процеси, не змінюючи збережений стан (при старті вони продовжаться). |
| 11 | `downloads()` — усі записи, найновіші (`started`) першими: `{repo, revision, user, status, done_gib, total_gib, percent, started, finished, error_code, error}`. |

### 5.2 Процес завантаження

Запуск: `spawn([python, "-m", "gpu_manager.download_worker"], stdin=PIPE, stdout=<лог>, stderr=<лог>, env=…)`;
у `stdin` пишеться JSON `{"repo", "revision", "files": [імена]}` і закривається. Оточення: `HF_HOME=<hf_home>`,
`HF_TOKEN` — лише якщо токен є. Лог — `<data_dir>/downloads/models--<org>--<name>.log`; кожна спроба спершу
дописує в нього рядок-маркер `=== gpu-manager download`, і причину невдачі шукають лише після останнього маркера
(помилка минулої спроби не повторюється в новій). При помилці процес
пише останнім рядком `ERROR <клас винятку>: <текст>` і виходить з кодом ≠ 0. Відповідність класу коду:
`GatedRepoError` → `hf_gated`; `RepositoryNotFoundError`, `RevisionNotFoundError` → `hf_not_found`;
«No space left» у лозі → `disk_full_during`; інше → `download_failed`.

### 5.3 Прогрес

Кеш HF: `<hf_home>/hub/models--<org>--<name>/snapshots/<sha>/<файл>` (готові файли) і
`…/blobs/*.incomplete` (недокачані). `done` = сума розмірів (з API) наявних у знімку файлів + сума розмірів
`*.incomplete`, не більше загального; `percent` = `100·done/total` (1 знак); у стані `done` — 100.

### 5.4 Список і видалення

| # | Правило |
|---|---|
| 1 | `local()`: моделі (не датасети) з `<hf_home>/hub`: `{repo, size_gib, revisions, revision, state, last_modified}`; `state` = `ready` (немає запису, запис `done` або є повна ревізія, яку менеджер докачав раніше, — навіть коли нова ревізія ще качається чи не докачалась), `downloading` (активний), `partial` (інше). `revision` — яку запускати: докачана менеджером (остання повна) → на яку вказує `refs/main` → найновіша. Немає теки — `[]`. |
| 2 | `delete(repo, user)`: активне завантаження → `download_active`; моделі немає на диску → `model_not_local`; інакше видаляються всі ревізії разом із блобами, тека моделі зникає повністю, запис черги видаляється; журнал `model_delete` (`freed_gib`); відповідь `{deleted: true, repo, freed_gib}`. |
| 3 | `info(repo, revision=None, fraction=None)`: `{repo, revision, gated, files, size_gib, fraction, fit (Fit як словник), local}`; `fraction` поза (0, 1] → `bad_fraction`. |
| 4 | `search(query, limit=10)`: `limit` обмежується до 1..50. |

## 6. HTTP (захист і формат відмов — SPEC-GPU-001 §7)

| # | Метод, шлях | Вхід | Вихід |
|---|---|---|---|
| 1 | `GET /api/models/search?q=&limit=` | | як `search` |
| 2 | `GET /api/models/info?repo=&revision=&fraction=` | | як `info` |
| 3 | `GET /api/models` | | `{local: [...], downloads: [...]}` |
| 4 | `POST /api/models/download` | `{repo, user, revision?}` | стан завантаження |
| 5 | `POST /api/models/cancel` | `{repo, user}` | стан завантаження |
| 6 | `POST /api/models/delete` | `{repo, user}` | `{deleted, repo, freed_gib}` |
| 7 | `GET /api/version` | | `{version, commit, date, dirty}`; `version` = `gpu_manager.__version__` (число версії не фіксується тестами: його піднімає людина); без комітів `commit`, `date`, `dirty` — `null` |

`build_app(cfg, manager, models=None)`: маршрути 1–6 і MCP-інструменти §7 з'являються лише з `models`.
Маршрути 1–6 не блокують сервер: поки чекають на HF чи диск, решта запитів (напр. `/api/overview`) відповідає.

## 7. MCP (формат — SPEC-GPU-001 §8; реєструються, лише якщо `build_mcp(..., models=store)`)

| # | Інструмент | Параметри | Результат |
|---|---|---|---|
| 1 | `hf_search` | `query, limit=10` | `[{repo, downloads, gated, params, task}]` |
| 2 | `hf_model` | `repo, revision?, fraction?` | як `info` |
| 3 | `model_download` | `repo, user, revision?` | `{repo, status, percent, gib: [done, total], user, started, error?}` |
| 4 | `model_downloads` | — | список того ж формату |
| 5 | `model_download_cancel` | `repo, user` | той самий формат |
| 6 | `models_local` | — | `[{repo, gib, state}]` |
| 7 | `model_delete` | `repo, user` | `{deleted, repo, freed_gib}`; анотація `destructiveHint: true` |

## 8. Нові коди відмов

`hf_unavailable`, `hf_not_found`, `hf_gated`, `bad_fraction`, `disk_full`, `download_active`, `download_not_active`,
`model_not_local`, `no_vllm_weights`; у стані завантаження також `disk_full`, `disk_full_during`, `download_failed`.

## 9. Уточнення (відповіді на прогалини тестів)

- `store` = `manager.store`, `journal` = `manager.journal_log` (той самий `state.json` і журнал, що в карт).
- `ManagerError(code, **params)` з `gpu_manager.messages`; поле `.code`. Відсутній параметр не валить показ тексту.
- `check_access` нічого не повертає; немає доступу → `ManagerError` (`hf_gated` / `hf_not_found` / `hf_unavailable`).
- `*_gib` — 2 знаки після коми; `started`, `finished` — unix-секунди (у MCP — рядок `YYYY-MM-DD HH:MM`).
- У stdin процесу: `revision` = sha з `hub.files`, `files` = усі вибрані файли.
- `cancel` і `delete` перевіряють `unknown_user`; будь-який дозволений користувач може скасувати чи видалити (журнал).
- Сервіс викликає `poll()` кожні `gpu.sample_interval_s`; при зупинці — `stop_all()`.
- Немає `repo` у запиті HTTP → `bad_request`.

## 10. Шви для тестів

Тести не ходять у мережу й не качають справжніх моделей: `hub` — підробка з методами §4; `spawn` — підробка,
що повертає об'єкт з `stdin` (`write`, `close`), `poll()`, `terminate()`, `wait(timeout)`, `kill()`; кеш HF —
тека в `tmp_path` з розкладкою §5.3 (файли знімка можна створювати звичайними файлами); `clock` — керований.
Живий тест з мережею позначається і запускається лише за `GPU_MANAGER_LIVE=1`.
