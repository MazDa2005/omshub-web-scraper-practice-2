#!/usr/bin/env python3
"""
Практическое занятие 3. Сбор данных через открытый API.
Дисциплина: «Сбор первичных данных цифрового следа».

Источник: GitHub GraphQL API (https://docs.github.com/en/graphql)
Эндпоинт: POST https://api.github.com/graphql
Документация поиска: https://docs.github.com/en/graphql/reference/queries#searchresultitemconnection

Связь с ПЗ1/ПЗ2: кейс Georgia Institute of Technology / OMSCS.
В ПЗ1 источником уже был GitHub (репозиторий api-evangelist/georgia-institute-of-technology),
в ПЗ2 — независимый сайт отзывов на курсы OMSCS (omshub.org). Здесь собираются открытые
репозитории студентов/сообщества, связанные с курсами OMSCS (решения, конспекты, автограйдеры),
по тем же кодам курсов (CS-6200, CS-6515 и т.п.), что встречались в датасете ПЗ2.

Пагинация: курсорная (GraphQL Relay cursor connections) — поле pageInfo { hasNextPage endCursor },
запрос продолжается с параметром after: <endCursor> до hasNextPage == false.

Запуск:
    pip install requests pandas python-dotenv
    echo "GITHUB_TOKEN=ghp_xxx" > .env     # .env добавлен в .gitignore
    python api_collector.py --query "OMSCS" --max-pages 10
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv
    load_dotenv()  # читает .env, который НЕ попадает в репозиторий (см. .gitignore)
except ImportError:
    pass  # python-dotenv опционален: ключ можно задать просто переменной окружения

API_URL = "https://api.github.com/graphql"
PAGE_SIZE = 50                 # записей за один запрос (GitHub допускает до 100)
SOFT_PAUSE = 0.3               # мягкая пауза между запросами сверх retry-политики (лекция 3, листинг 3)
TIMEOUT = 15

OUT_DIR = Path(".")
LOG_FILE = OUT_DIR / "api_scrape.log"
CSV_FILE = OUT_DIR / "github_repos_dataset.csv"
CHECKPOINT = OUT_DIR / "checkpoint_api.jsonl"
STATS_FILE = OUT_DIR / "api_dataset_stats.txt"

log = logging.getLogger("api_collector")

# GraphQL-запрос: поиск репозиториев, курсорная пагинация через $after
SEARCH_QUERY = """
query SearchRepos($q: String!, $first: Int!, $after: String) {
  rateLimit { limit cost remaining resetAt }
  search(query: $q, type: REPOSITORY, first: $first, after: $after) {
    repositoryCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Repository {
        id
        nameWithOwner
        description
        url
        isFork
        stargazerCount
        forkCount
        createdAt
        updatedAt
        primaryLanguage { name }
        owner { login }
        licenseInfo { spdxId }
      }
    }
  }
}
"""


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    for h in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def build_session(token: str) -> requests.Session:
    """requests.Session с HTTPAdapter + Retry: повтор при 429/5xx, экспоненциальный backoff (этап 4)."""
    retry = Retry(
        total=5,
        backoff_factor=1.5,                     # 1.5, 3, 6, 12, 24 сек.
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],                # GraphQL всегда POST, в отличие от REST-листинга лекции
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "mirea-digital-footprint-practice-3",  # GitHub API требует непустой User-Agent
    })
    return session


def graphql_request(session: requests.Session, query: str, variables: dict) -> dict:
    """Один запрос к GraphQL. Логирует код ответа и остаток лимита, бросает исключение на явной ошибке."""
    resp = session.post(API_URL, json={"query": query, "variables": variables}, timeout=TIMEOUT)
    log.info("HTTP %s | попытка после retry (если были 429/5xx — см. выше)", resp.status_code)

    if resp.status_code != 200:
        # Этап 6: нештатная ситуация, не покрытая автоматическим retry (например, 401 — неверный токен)
        log.error("Неожиданный статус %s, тело: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()

    payload = resp.json()

    if "errors" in payload:
        # GraphQL может вернуть 200 с ошибкой в теле (например, превышен "cost" бюджет запроса)
        log.error("GraphQL ошибка в теле ответа: %s", payload["errors"])
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")

    rl = payload["data"].get("rateLimit", {})
    if rl:
        remaining, limit = rl.get("remaining"), rl.get("limit")
        log.info("Лимит GraphQL: остаток %s из %s (cost запроса: %s, сброс: %s)",
                  remaining, limit, rl.get("cost"), rl.get("resetAt"))
        if limit and remaining is not None and remaining / limit < 0.05:
            # Этап 6: заранее логируем приближение к лимиту и подстраховываемся паузой
            log.warning("Остаток лимита < 5%% — увеличиваю паузу между запросами")
            time.sleep(5)

    return payload["data"]


def load_checkpoint():
    rows, done_cursor = [], None
    if CHECKPOINT.exists():
        lines = CHECKPOINT.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                rec = json.loads(line)
                if rec.get("_cursor_marker"):
                    done_cursor = rec["_cursor_marker"]
                else:
                    rows.append(rec)
    return rows, done_cursor


def fetch_all(session: requests.Session, query_text: str, max_pages: int, fresh: bool) -> tuple[list[dict], list[str]]:
    """Курсорная пагинация (этап 5): цикл продолжается, пока search.pageInfo.hasNextPage == true."""
    if fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        log.info("Чекпоинт удалён (--fresh)")

    records, cursor = load_checkpoint()
    if cursor:
        log.info("Найден чекпоинт: %d записей, продолжаем с cursor=%s...", len(records), cursor[:16])

    errors: list[str] = []
    page_no = len(records) // PAGE_SIZE

    while page_no < max_pages:
        variables = {"q": query_text, "first": PAGE_SIZE, "after": cursor}
        try:
            data = graphql_request(session, SEARCH_QUERY, variables)
        except (requests.RequestException, RuntimeError) as exc:
            # Этап 6: сетевой таймаут / превышение попыток retry / ошибка схемы — не теряем уже собранное
            log.error("Остановка после ошибки на странице %d: %s. Собрано к этому моменту: %d записей",
                      page_no + 1, exc, len(records))
            errors.append(str(exc))
            break

        search = data["search"]
        nodes = search["nodes"]
        if not nodes:
            log.info("Страница %d: пустой результат — данные закончились", page_no + 1)
            break

        records.extend(nodes)
        with CHECKPOINT.open("a", encoding="utf-8") as f:
            for n in nodes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")

        page_info = search["pageInfo"]
        cursor = page_info["endCursor"]
        with CHECKPOINT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"_cursor_marker": cursor}, ensure_ascii=False) + "\n")

        log.info("Страница %d обработана, накоплено записей: %d (hasNextPage=%s, repositoryCount=%s)",
                  page_no + 1, len(records), page_info["hasNextPage"], search["repositoryCount"])

        page_no += 1
        if not page_info["hasNextPage"]:
            log.info("hasNextPage=false — обход завершён, все страницы получены")
            break
        time.sleep(SOFT_PAUSE)

    return records, errors


def build_dataset(records: list[dict]) -> pd.DataFrame:
    """Этап 7: json_normalize разворачивает вложенные объекты (owner, primaryLanguage, licenseInfo)."""
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records, sep="_")
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    # Извлекаем код курса OMSCS из названия/описания репозитория регулярным выражением,
    # чтобы потом сопоставить с course_id из датасета ПЗ2 (omshub_reviews_dataset.csv)
    text = (df.get("nameWithOwner", "").fillna("") + " " + df.get("description", "").fillna(""))
    extracted = text.str.upper().str.extract(r"CS[-\s]?(\d{4})")[0]
    df["course_code"] = extracted.apply(lambda d: f"CS-{d}" if pd.notna(d) else None)

    df["scraped_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return df


def compare_with_pz2(df_api: pd.DataFrame, pz2_path: Path) -> list[str]:
    """Этап 8: сопоставление с датасетом ПЗ2 по коду курса."""
    if not pz2_path.exists():
        return [f"Файл ПЗ2 не найден по пути {pz2_path} — сопоставление пропущено, заполните вручную"]

    df2 = pd.read_csv(pz2_path)
    notes = []
    if "course_id" not in df2.columns:
        return [f"В {pz2_path} нет колонки course_id — проверьте имя файла/структуру"]

    courses_reviews = set(df2["course_id"].dropna().str.upper())
    courses_repos = set(df_api["course_code"].dropna())
    only_reviews = courses_reviews - courses_repos
    only_repos = courses_repos - courses_reviews
    both = courses_reviews & courses_repos

    notes.append(f"Курсов с отзывами (ПЗ2): {len(courses_reviews)}; курсов с упоминанием в репозиториях (ПЗ3): {len(courses_repos)}")
    notes.append(f"Есть и отзывы, и репозитории: {sorted(both)}")
    notes.append(f"Только отзывы, репозиториев не найдено: {sorted(only_reviews)}")
    notes.append(f"Только репозитории, отзывов не найдено (в собранном срезе): {sorted(only_repos)}")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="OMSCS Georgia Tech in:name,description,readme",
                     help="поисковый запрос GitHub (синтаксис https://docs.github.com/search-syntax)")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--pz2-csv", default="omshub_reviews_dataset.csv",
                     help="путь к датасету ПЗ2 для сопоставления (этап 8)")
    args = ap.parse_args()

    setup_logging()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        # Листинг 1 из задания: сбой должен быть явным, а не тихим
        log.error("Переменная окружения GITHUB_TOKEN не задана. Создайте .env (см. .env.example) и повторите запуск.")
        raise SystemExit(1)

    log.info("=== Старт сбора: GitHub GraphQL, запрос=%r, лимит страниц=%d ===", args.query, args.max_pages)
    t0 = time.time()
    session = build_session(token)
    records, errors = fetch_all(session, args.query, args.max_pages, args.fresh)
    df = build_dataset(records)

    if df.empty:
        log.error("Датасет пуст — проверьте запрос, токен и лог")
        return

    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    comparison_notes = compare_with_pz2(df, Path(args.pz2_csv))

    lines = [
        f"Источник: GitHub GraphQL API ({API_URL})",
        f"Поисковый запрос: {args.query}",
        f"Дата и время сбора (UTC): {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')}",
        f"Собрано страниц: {df.shape[0] // PAGE_SIZE + 1}, записей: {len(df)}, колонок: {df.shape[1]}",
        f"Ошибки за время сбора: {errors if errors else 'нет'}",
        "",
        "Колонки датасета:",
        ", ".join(df.columns),
        "",
        "Пропуски по ключевым полям:",
        df[["nameWithOwner", "description", "stargazerCount", "course_code"]].isna().sum().to_string(),
        "",
        f"Репозиториев с извлечённым кодом курса (course_code): {df['course_code'].notna().sum()} из {len(df)}",
        "",
        "Сопоставление с ПЗ2:",
        *[f"  - {n}" for n in comparison_notes],
    ]
    text = "\n".join(lines)
    STATS_FILE.write_text(text, encoding="utf-8")
    print("\n" + text)
    log.info("=== Готово: %d записей за %.0f с. Файл: %s ===", len(df), time.time() - t0, CSV_FILE.name)


if __name__ == "__main__":
    main()