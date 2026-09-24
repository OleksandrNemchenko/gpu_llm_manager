---
id: SPEC-GPU-001
status: draft
owner: gpu-manager
verified: 2026-09-23 21:06
---

# Специфікація фази 1: карти, бронювання, вебсторінка, MCP

Джерело — обговорення з користувачем 2026-09-23. Це контракт поведінки й інтерфейсів; реалізація —
`gpu_manager/`. Тести пишуться лише з цього документа.

## 1. Призначення

Один сервіс на сервері з кількома GPU NVIDIA (кількість і модель карт — з NVML, у коді не зашиваються). Показує стан кожної карти й дозволяє 4 довіреним користувачам
позначати карту як зайняту чи вільну. Два входи з однаковою поведінкою: вебсторінка (для людини,
українською) і MCP (для агентів, англійською).

## 2. Конфіг `config.json`

Файл редагує людина; сервіс читає його при старті.

| # | Правило |
|---|---|
| 1 | Верхній рівень — об'єкт розділів. Розділ — об'єкт, у якому ключ `comment` (рядок) описує розділ, а решта ключів — налаштування або вкладені розділи. |
| 2 | Налаштування — об'єкт рівно з двома ключами: `value` і `comment`; `comment` — непорожній рядок (число чи інший тип → `ConfigError`). |
| 3 | Налаштування без обгортки (голе значення), зайвий ключ або порожній `comment` → `ConfigError`, у тексті — шлях налаштування (напр. `server.port`). |
| 4 | Обов'язкові налаштування: `server.hosts` (список адрес), `server.port`, `server.display_timezone` (IANA, напр. `Europe/Kyiv`), `users.allowed` (список логінів), `gpu.sample_interval_s`, `gpu.ring_keep_s`, `gpu.history_db_interval_s`, `gpu.history_db_keep_days`, `gpu.busy_memory_mib`, `vllm.port_range` (`[від, до]` включно), `journal.keep_entries`, `journal.keep_days`, `paths.data_dir`. Необов'язкові: `server.extra_host_names` (список), `files.inbox_dir` (шлях, типово `/srv/gpu-inbox`). Відсутнє обов'язкове → `ConfigError`. |
| 5 | `server.hosts` з `0.0.0.0`, `::` або порожнім рядком → `ConfigError`. Порожній `server.hosts` або `users.allowed` → `ConfigError`. |
| 6 | `server.port` поза 1024..65535 → `ConfigError`. `vllm.port_range` поза 1024..65535, з `від > до`, або такий, що містить `server.port` → `ConfigError`. |
| 7 | `gpu.sample_interval_s ≤ 0`, `journal.keep_entries ≤ 0` або `journal.keep_days ≤ 0` → `ConfigError`. |
| 8 | Відносний `paths.data_dir` рахується від теки, де лежить конфіг. |
| 9 | Дозволені імена хоста (`host_names`) = `server.hosts` + `server.extra_host_names`, без повторів, у цьому порядку. |

Python: `gpu_manager.config.load_config(path: Path) -> Config`, `gpu_manager.config.ConfigError`.
Поля `Config`: `hosts`, `port`, `host_names`, `display_timezone` (`ZoneInfo`), `users`, `sample_interval_s`,
`ring_keep_s`, `history_db_interval_s`, `history_db_keep_days`, `busy_memory_mib`, `model_ports`
(кортеж `(від, до)`), `journal_keep_entries`, `journal_keep_days`, `data_dir` (`Path`).

## 3. Стан карти

| # | Статус | Коли |
|---|---|---|
| 1 | `reserved` | є бронювання, строк не минув (або строку немає) |
| 2 | `reserved_expired` | є бронювання, момент `until` настав (`now ≥ until`) |
| 3 | `unknown` | бронювання немає, а телеметрії карти немає або вона з помилкою |
| 4 | `busy` | бронювання немає, але є процеси на карті або зайнята пам'ять ≥ `busy_memory_mib` |
| 5 | `free` | усе інше |

Пріоритет — у порядку рядків: бронювання важливіше за процеси.

Прострочене бронювання **не знімається автоматично** — лише людиною.

## 4. Бронювання

| # | Правило |
|---|---|
| 1 | `user` має бути з `users.allowed`, інакше відмова `unknown_user`. |
| 2 | `gpu` — ціле 0..N-1 (N — кількість карт); інше (зокрема `true`/`false`) → `unknown_gpu`. |
| 3 | Бронювання вільної карти: запис `{gpu, user, purpose, since=now, until}`; `until = now + hours·3600` або `null`, якщо `hours` не задано. |
| 4 | `hours ≤ 0` → `bad_hours`. |
| 5 | Повторне бронювання тим самим користувачем: оновлює `purpose` і `until`, **`since` не змінюється**. |
| 6 | Бронювання чужої заброньованої карти → `reserved_by_other`; бронювання не змінюється. |
| 7 | Якщо на карті вже є процеси, власник яких ≠ `user`, бронювання відбувається, але відповідь містить попередження `foreign_processes`. |
| 8 | Звільнення власного бронювання — успіх, `released: true`. |
| 9 | Звільнення незаброньованої карти — не помилка: `released: false`. |
| 10 | Звільнення чужого бронювання без `force` → `release_needs_force`, бронювання лишається. З `force=true` — знімається. |
| 11 | `purpose` зберігається з обрізаними пробілами на краях. |
| 12 | Бронювання переживають перезапуск сервісу (зберігаються в `data_dir/state.json`). |
| 13 | Пошкоджений `state.json` (не JSON або не об'єкт) — сервіс **не стартує** (`StateError`), а не починає з порожнього стану. |

## 5. Журнал дій

Кожна успішна зміна пишеться в журнал: `{ts, user, action, gpu, …}`.

| # | Дія | `action` | Додаткові поля |
|---|---|---|---|
| 1 | нове бронювання | `reserve` | `purpose`, `until` |
| 2 | повторне бронювання тим самим | `reserve_update` | `purpose`, `until` |
| 3 | звільнення свого | `release` | — |
| 4 | звільнення чужого з force | `release_forced` | `owner` (чиє було), `purpose` |

Відмови й `released: false` у журнал не пишуться. Журнал повертається від найновішого запису. Файл — `data_dir/journal.jsonl`;
рядок, обірваний аварійною зупинкою, не має ховати ні попередні записи, ні наступні.

Зберігання обмежене двома межами з конфігу: лишаються лише останні `journal.keep_entries` записів і лише
ті, що молодші за `journal.keep_days` днів (`ts ≥ now − keep_days·86400`); запис поза будь-якою межею
видаляється і більше не повертається.

## 6. Історія телеметрії

- Щосекундні точки зберігаються в пам'яті `ring_keep_s` секунд; кожні `history_db_interval_s` секунд у SQLite
  (`data_dir/history.sqlite`) пишеться зведена точка.
- Запит: карта (або всі), період у хвилинах, максимум точок на карту. Період ≤ `ring_keep_s` — з пам'яті,
  довший — із SQLite.
- Відповідь на карту — колонки однакової довжини: `t` (unix-секунди), `temp_c`, `util_pct`, `power_w`, `mem_mib`;
  не більше заданої кількості точок.
- Проріджування: температура й пам'ять — **максимум** у групі (пік перегріву не губиться); завантаження
  й потужність — середнє.
- Заміри з помилкою в історію не потрапляють.
- `minutes ≤ 0` або `points ≤ 0` → `bad_history`.

## 7. Вебсторінка та HTTP API

Порт — `server.port`. Сервіс слухає лише адреси з `server.hosts`.

### 7.1 Захист (автентифікації немає — користувачі довірені)

| # | Правило | Відповідь |
|---|---|---|
| 1 | Заголовок `Host` не з `host_names` (з портом або без) | 421 |
| 2 | `POST /api/*` з `Content-Type`, що не `application/json` | 415 |
| 3 | `POST /api/*` із заголовком `Origin`, що не `http://<дозволений host[:port]>` | 403 |
| 4 | `POST` без `Origin` дозволено (агенти, curl) | — |

### 7.2 Маршрути

| # | Метод, шлях | Вхід | Вихід |
|---|---|---|---|
| 1 | `GET /` | — | HTML-сторінка |
| 2 | `GET /healthz` | — | `{"ok": true}`, тіло закінчується символом `\n` |
| 3 | `GET /api/overview` | — | `{ts, users, public_host, gpus: [карта…]}` |
| 4 | `GET /api/journal?limit=N` | `limit` (типово 30) | `[запис…]`, найновіший перший |
| 5 | `GET /api/history?gpu=&minutes=&points=` | усі необов'язкові (типово всі карти, 60 хв, 120 точок) | `{"<gpu>": {t, temp_c, util_pct, power_w, mem_mib}}` |
| 6 | `POST /api/reserve` | `{gpu, user, purpose?, hours?}`; `hours` — число, рядок-число, `""` або `null` (без строку) | `{reserved: true, gpu: карта, warnings: [текст…]}` |
| 7 | `POST /api/release` | `{gpu, user, force?}`; примусово лише при `force: true` (саме JSON true) | `{released, previous, gpu: карта}`; `previous` — зняте бронювання або `null` |

Карта в `gpus`: `index`, `name`, `uuid`, `pci_bus_id`, `memory_total_mib`, `power_limit_w`, `ecc_enabled`,
`compute_capability`, `temp_slowdown_c`, `temperature_c`, `util_pct`, `power_w`, `memory_used_mib`, `error`,
`status`, `users` (відсортовані логіни власників процесів + власник бронювання), `processes`
(`[{pid, user, name, cmdline, used_mib, kind}]`), `reservation` (`{gpu, user, purpose, since, until, expired}`
або `null`).

`public_host` — перша адреса з `server.hosts`, що не loopback (уся `127.0.0.0/8`, `::1`, `localhost` — RFC 1122), з портом (`"203.0.113.7:1200"`).

Відмова — HTTP 400, тіло `{"code": "<код>", "error": "<текст українською>"}`. Некоректний JSON або
відсутнє обов'язкове поле — код `bad_request`.

### 7.3 Сторінка

Українською, мінімум тексту, зручна на телефоні, тема — за налаштуванням системи. Картка кожної карти:
температура, завантаження, потужність, пам'ять, процеси (власник, пам'ять; командний рядок — на дотик),
бронювання. Зайнята карта — вся картка іншого кольору. Вибір «хто я» зберігається в cookie. Внизу —
згорнуті блоки «Як користуватися» (коротка інструкція) і «Підключити агента (MCP)» (команди для Claude Code
і Codex з уже підставленими адресою та логіном).
Оновлення — щосекунди, поки вкладка видима.

## 8. MCP

Streamable HTTP за шляхом `/mcp`, той самий порт. Без сесій (перезапуск сервісу непомітний клієнтам).
Результат кожного інструмента — **один текстовий блок з JSON без переносів рядків** (економія токенів агента).
Тексти — англійською. Час — рядок `YYYY-MM-DD HH:MM` у поясі `display_timezone`; поле `tz` — назва поясу.

| # | Інструмент | Параметри | Результат |
|---|---|---|---|
| 1 | `gpu_status` | `gpu?` | `{tz, gpus: [коротка карта]}` (одна, якщо задано `gpu`) |
| 2 | `gpu_free` | — | `[{gpu, mem_total_mib, temp_c}]` лише для статусу `free` |
| 3 | `gpu_who` | `gpu` | `{tz, …коротка карта з cmd у процесах, users}` |
| 4 | `gpu_reserve` | `gpu, user, purpose="", hours?` | `{reserved: true, gpu: коротка карта, warnings?: [текст]}` |
| 5 | `gpu_release` | `gpu, user, force=false` | `{released, previous_owner, gpu: коротка карта}` |
| 6 | `gpu_history` | `gpu?, minutes=60, points=60` | `{tz, series: {"<gpu>": колонки}}` |
| 7 | `gpu_journal` | `limit=20` | `[запис з ts рядком часу]` |
| 8 | `gpu_guide` | — | Markdown-інструкція для агентів (не JSON): логіни з `users.allowed`, `files.inbox_dir`, адреса `public_host` без порту, стан теки (`Status: ready.`, якщо тека існує, інакше `Status: NOT set up …`). |

Коротка карта: `{gpu, status, temp_c, util_pct, power_w, mem_mib: [used, total], reservation: {user, purpose,
since, until, expired} | null, processes: [{pid, user, name, mib}], error?}`.

Відмова — результат з `isError: true`; текст містить `"<code>: <English text>"` (SDK може додати свій префікс перед ним).

## 9. Коди відмов і попереджень

| # | Код | Коли |
|---|---|---|
| 1 | `unknown_gpu` | немає такої карти |
| 2 | `unknown_user` | логін не з `users.allowed` |
| 3 | `reserved_by_other` | бронювання чужої заброньованої карти |
| 4 | `release_needs_force` | звільнення чужого без `force` |
| 5 | `bad_hours` | `hours ≤ 0` |
| 6 | `bad_history` | `minutes ≤ 0` або `points ≤ 0` |
| 7 | `bad_request` | некоректне тіло HTTP-запиту |
| 8 | `foreign_processes` | попередження: на заброньованій карті процеси інших власників |

## 10. Шви для тестів (Python)

Тести не мають торкатися справжніх карт і справжнього часу.

- `gpu_manager.gpu`: dataclass-и `GpuInfo(index, name, uuid, pci_bus_id, memory_total_mib, power_limit_w,
  ecc_enabled, compute_capability, temp_slowdown_c)`, `GpuSample(index, ts, temperature_c, util_pct, power_w,
  memory_used_mib, error=None)`, `GpuProcess(pid, user, name, cmdline, used_mib, kind)`; протокол `GpuBackend`
  з методами `info() -> list[GpuInfo]`, `sample(ts) -> list[GpuSample]`, `processes() -> dict[int, list[GpuProcess]]`.
  Справжній бекенд — `gpu_manager.gpu.NvmlBackend()` (без аргументів; читає всі карти через NVML).
- `gpu_manager.app.build_manager(cfg, backend, clock=time.time) -> GpuManager` — `clock` повертає поточний
  unix-час; усі строки й історія рахуються від нього.
- `GpuManager.tick()` — один крок опитування (читає `backend.sample` і `backend.processes`).
- `gpu_manager.app.build_app(cfg, manager) -> Starlette` — ASGI-застосунок; під час старту (lifespan) сам
  робить `tick()` і далі опитує кожні `sample_interval_s`.
- `gpu_manager.mcp_tools.build_mcp(manager, tz, guide=None) -> MCPServer`; `guide` — функція без аргументів, що
  повертає текст для `gpu_guide` (без неї — `"No guide configured."`); `gpu_manager.app.render_guide(cfg) -> str` (`mcp.server.mcpserver.MCPServer`, mcp 2.2.0) — для
  виклику інструментів у процесі (асинхронно, напр. через `anyio.run`):
  - `await mcp.list_tools()` → список об'єктів з полем `.name`;
  - `await mcp.call_tool(name, args)` → результат; JSON-рядок відповіді — `result.content[0].text`;
  - відмова в процесі **піднімає** `mcp.server.mcpserver.exceptions.ToolError`; `str(exc)` містить
    `"<code>: <English text>"`. Поле `isError: true` бачить лише клієнт по HTTP.
- Відмови в Python — `gpu_manager.messages.ManagerError` з полем `code`.
- `gpu_manager.gpu.describe_process(pid, proc_root="/proc") -> (owner, name, cmdline)`: читає
  `<proc_root>/<pid>/comm` і `cmdline` (аргументи через `\0` → пробіли, обрізка до 300 символів); власник —
  логін власника теки `<proc_root>/<pid>`; неіснуючий pid → `(None, "?", "")`.

Правило GPU для прогону тестів: `CUDA_VISIBLE_DEVICES=2,3`. Тест, що читає справжній NVML, позначається
і запускається лише за змінної оточення `GPU_MANAGER_LIVE=1`. Він звіряє `NvmlBackend().info()` з
`nvidia-smi --query-gpu=index,name --format=csv,noheader`: ті самі індекси й назви карт; число й модель карт
тест не зашиває.
