#!/usr/bin/env python3


import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------- настройки ------------------------------------

BASE_URL = "https://omshub.org/"
USER_AGENT = "Mozilla/5.0 (educational scraper; MIREA practice 2)"   # только ASCII
MIN_DELAY = 2.0            # секунд между запросами (лекция 2)
TIMEOUT = 10
DEFAULT_MAX_COURSES = 40
# Запрещено robots.txt сайта (Disallow) — дополнительная страховка поверх robotparser
BLOCKED_PREFIXES = ("/api/", "/user/")

# Запасной список, если ссылки на курсы не нашлись в HTML главной страницы.
# Несуществующий ID даст 404 — скрипт запишет это в лог и пойдёт дальше.
SEED_COURSES = [
    "CS-6200", "CS-6515", "CS-6310", "CS-7641", "CS-6601", "CS-7642", "CS-7637", "CS-6750",
    "CS-6035", "CS-6250", "CS-6400", "CS-7646", "CS-6476", "CS-7643", "CS-7650", "CS-6300",
    "CS-6290", "CS-7638", "CS-6340", "CS-6262",
]

OUT_DIR = Path(".")
CHECKPOINT = OUT_DIR / "checkpoint_omshub.jsonl"
LOG_FILE = OUT_DIR / "scrape.log"
ROBOTS_COPY = OUT_DIR / "robots_txt_copy.txt"
CSV_FILE = OUT_DIR / "omshub_reviews_dataset.csv"
JSON_FILE = OUT_DIR / "omshub_reviews_dataset.json"
SUMMARY_FILE = OUT_DIR / "course_summary.csv"
STATS_FILE = OUT_DIR / "dataset_stats.txt"
HEAD_FILE = OUT_DIR / "first_15_rows.csv"

# CSS-селекторы (этап 2). Хэшированные классы Mantine (m_347db0ec...) меняются при пересборке
# сайта, поэтому опираемся на data-атрибуты и статические классы mantine-*.
SEL_CARD = 'div[data-review-card="true"]'
SEL_BADGE_LABEL = "span.mantine-Badge-label"     # 1-й = семестр; далее «Verified»/«Legacy»
SEL_COURSE = 'p[data-size="md"]'                 # «CS-6310: Software Architecture and Design»
SEL_SMALL = 'p[data-size="xs"]'                  # дата отзыва — первый p с датой
SEL_STAT = 'p[data-size="lg"]'                   # workload, difficulty, overall (по порядку)
SEL_BODY = "div.review-body"                     # берём только длину, не текст
SEL_TOTAL = 'p[data-size="sm"]'                  # «20 of 163 reviews (load more below)»
SEL_COURSE_LINK = 'a[href^="/course/"]'

SEMESTER_RE = re.compile(r"(SPRING|SUMMER|FALL)\s+(\d{4})", re.I)
COURSE_RE = re.compile(r"^\s*([A-Z]+-\d+[A-Z]?)\s*:\s*(.+?)\s*$")
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
TOTAL_RE = re.compile(r"(\d+)\s+of\s+(\d+)\s+review", re.I)
COURSE_ID_RE = re.compile(r"^/course/([A-Za-z]+-\d+[A-Za-z]?)/?$")

log = logging.getLogger("scraper")


# ----------------------------- вспомогательное ------------------------------

def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    for h in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en,ru;q=0.8"})
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET", "HEAD"], respect_retry_after_header=True)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def check_robots(session: requests.Session, base: str):
    robots_url = urljoin(base, "robots.txt")
    rp = robotparser.RobotFileParser()
    try:
        r = session.get(robots_url, timeout=TIMEOUT)
        stamp = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
        ROBOTS_COPY.write_text(
            f"URL: {robots_url}\nДата проверки: {stamp}\nHTTP-статус: {r.status_code}\n{'-' * 40}\n"
            f"{r.text if r.status_code == 200 else '(файл отсутствует или недоступен)'}\n", encoding="utf-8")
        log.info("robots.txt: %s -> HTTP %s", robots_url, r.status_code)
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
    except requests.RequestException as exc:
        log.warning("robots.txt не получен (%s) — работаем с задержкой по умолчанию", exc)
        rp.parse([])
    return rp, rp.crawl_delay("*")


def allowed(rp: robotparser.RobotFileParser, url: str) -> bool:
    path = urlparse(url).path
    if path.startswith(BLOCKED_PREFIXES):
        return False
    return rp.can_fetch(USER_AGENT, url)


def polite_sleep(delay: float) -> None:
    time.sleep(delay + random.uniform(0, 0.5))


def first_int(text: str):
    m = re.search(r"\d+", text or "")
    return int(m.group()) if m else None


# ----------------------------- разбор страниц -------------------------------

def parse_course_links(html: str) -> list[str]:
    """Достаёт ID курсов из ссылок /course/<ID> на главной странице."""
    soup = BeautifulSoup(html, "html.parser")
    ids, seen = [], set()
    for a in soup.select(SEL_COURSE_LINK):
        m = COURSE_ID_RE.match(a.get("href", ""))
        if m and m.group(1).upper() not in seen:
            seen.add(m.group(1).upper())
            ids.append(m.group(1).upper())
    return ids


def parse_course_page(html: str, page_url: str, page_no: int) -> list[dict]:
    """Разбирает страницу курса: только структурные поля отзывов."""
    soup = BeautifulSoup(html, "html.parser")

    total_on_site = None
    for p in soup.select(SEL_TOTAL):
        m = TOTAL_RE.search(p.get_text(" ", strip=True))
        if m:
            total_on_site = int(m.group(2))
            break

    rows = []
    for card in soup.select(SEL_CARD):
        labels = [s.get_text(strip=True) for s in card.select(SEL_BADGE_LABEL)]
        sem = SEMESTER_RE.search(labels[0]) if labels else None
        course = card.select_one(SEL_COURSE)
        cm = COURSE_RE.match(course.get_text(strip=True)) if course is not None else None

        review_date = None
        for p in card.select(SEL_SMALL):
            t = p.get_text(strip=True)
            if DATE_RE.match(t):
                try:
                    review_date = datetime.strptime(t, "%m/%d/%Y").date().isoformat()
                except ValueError:
                    review_date = None
                break

        stats = [first_int(p.get_text(strip=True)) for p in card.select(SEL_STAT)]
        stats += [None] * (3 - len(stats))
        body = card.select_one(SEL_BODY)
        rows.append({
            "course_id": cm.group(1) if cm else None,
            "course_name": cm.group(2) if cm else None,
            "semester": f"{sem.group(1).upper()} {sem.group(2)}" if sem else None,
            "term": sem.group(1).upper() if sem else None,
            "year": int(sem.group(2)) if sem else None,
            "review_date": review_date,
            "workload_hrs": stats[0],
            "difficulty": stats[1],
            "overall": stats[2],
            "verified_gt": "Verified" in labels[1:],
            "legacy": "Legacy" in labels[1:],
            # длина текста в словах — производная величина; сам текст не сохраняется
            "body_words": len(body.get_text(" ", strip=True).split()) if body is not None else None,
            "course_total_reviews": total_on_site,
            "source_url": page_url,
            "page": page_no,
        })
    return rows


# ----------------------------- сбор -----------------------------------------

def fetch(session, rp, url, delay):
    """Один вежливый GET. Возвращает Response или None."""
    if not allowed(rp, url):
        log.error("Запрещено robots.txt — пропуск: %s", url)
        return None
    try:
        return session.get(url, timeout=TIMEOUT)
    except requests.RequestException as exc:
        log.error("Сетевая ошибка после повторов: %s (%s)", url, exc)
        return None


def load_checkpoint():
    rows, done = [], set()
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add(r.get("_course"))
    return rows, done


def discover_courses(session, rp, base, delay) -> list[str]:
    for path in ("", "course"):
        url = urljoin(base, path)
        resp = fetch(session, rp, url, delay)
        polite_sleep(delay)
        if resp is not None and resp.status_code == 200:
            resp.encoding = "utf-8"
            ids = parse_course_links(resp.text)
            log.info("Список курсов: %s -> найдено ссылок: %d", url, len(ids))
            if len(ids) >= 3:
                return ids
    log.warning("Ссылки на курсы в HTML не найдены (страница рендерится JS?) — берём запасной список")
    return list(SEED_COURSES)


def scrape(base: str, courses_arg: list[str] | None, max_courses: int, fresh: bool):
    session = make_session()
    rp, crawl_delay = check_robots(session, base)
    delay = max(MIN_DELAY, float(crawl_delay or 0))
    log.info("Задержка между запросами: %.1f с (Crawl-delay=%s)", delay, crawl_delay)

    if fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    all_rows, done = load_checkpoint()
    if done:
        log.info("Чекпоинт: %d курсов, %d записей — продолжаем", len(done), len(all_rows))

    courses = courses_arg or discover_courses(session, rp, base, delay)
    courses = courses[:max_courses]
    log.info("К обходу: %d курсов", len(courses))

    failed, consecutive = [], 0
    for i, cid in enumerate(courses, 1):
        if cid in done:
            continue
        url = urljoin(base, f"course/{cid}")
        resp = fetch(session, rp, url, delay)
        if resp is None:
            failed.append(cid)
            consecutive += 1
        elif resp.status_code == 404:
            log.warning("[%d/%d] %s: HTTP 404 — такого курса нет, пропуск", i, len(courses), cid)
            consecutive = 0
        elif resp.status_code != 200:
            log.error("[%d/%d] %s: HTTP %s — пропуск", i, len(courses), cid, resp.status_code)
            failed.append(cid)
            consecutive += 1
        else:
            consecutive = 0
            resp.encoding = "utf-8"
            rows = parse_course_page(resp.text, url, i)
            if not rows:
                log.warning("[%d/%d] %s: карточек отзывов нет (изменилась вёрстка?)", i, len(courses), cid)
                failed.append(cid)
            else:
                for r in rows:
                    r["_course"] = cid
                with CHECKPOINT.open("a", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                all_rows.extend(rows)
                log.info("[%d/%d] %s: %d отзывов (всего %d)", i, len(courses), cid, len(rows), len(all_rows))
        if consecutive >= 3:
            log.error("3 ошибки подряд — останавливаемся, чтобы не нагружать сайт")
            break
        polite_sleep(delay)
    return all_rows, failed


# ----------------------------- очистка и проверка ---------------------------

def build_dataset(rows):
    notes = []
    df = pd.DataFrame(rows)
    if df.empty:
        return df, ["Данные не собраны"]
    df = df.drop(columns=["_course"], errors="ignore")
    df["scraped_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    key = ["course_id", "semester", "review_date", "workload_hrs", "difficulty", "overall", "body_words"]
    dup = int(df.duplicated(subset=key).sum())
    if dup:
        df = df.drop_duplicates(subset=key, keep="first")
        notes.append(f"Удалено полных дубликатов по ключу {key}: {dup}")
    else:
        notes.append("Полных дубликатов нет")

    notes.append(f"Оценки вне 1–5 (difficulty/overall): {int((~df['difficulty'].between(1, 5)).sum() + (~df['overall'].between(1, 5)).sum())}")
    notes.append(f"Некорректный workload (<=0 или >80 ч/нед): {int(((df['workload_hrs'] <= 0) | (df['workload_hrs'] > 80)).sum())}")
    notes.append(f"Записей без даты: {int(df['review_date'].isna().sum())}")
    notes.append(f"Записей с пометкой Legacy (перенесены со старой платформы): {int(df['legacy'].sum())}")
    notes.append(f"Записей с пометкой Verified GT email: {int(df['verified_gt'].sum())}")
    return df.reset_index(drop=True), notes


def write_outputs(df, notes, failed):
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    df.to_json(JSON_FILE, orient="records", force_ascii=False, indent=2)
    df.head(15)[["course_id", "semester", "review_date", "workload_hrs", "difficulty", "overall", "verified_gt"]] \
        .to_csv(HEAD_FILE, index=False, encoding="utf-8-sig")

    summary = (df.groupby(["course_id", "course_name"])
                 .agg(n_reviews=("overall", "size"), avg_workload=("workload_hrs", "mean"),
                      avg_difficulty=("difficulty", "mean"), avg_overall=("overall", "mean"),
                      total_on_site=("course_total_reviews", "max"))
                 .round(2).reset_index().sort_values("course_id"))
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    lines = [
        f"Источник: {BASE_URL}",
        f"Дата и время сбора (UTC): {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')}",
        f"Число строк: {len(df)}",
        f"Число курсов (страниц): {df['course_id'].nunique()}",
        f"Курсы с ошибками: {failed if failed else 'нет'}",
        "",
        "Пропуски по полям:",
        df.isna().sum().to_string(),
        "",
        "Числовые поля:",
        df[["workload_hrs", "difficulty", "overall", "body_words"]].describe().round(2).to_string(),
        "",
        "Проверки качества:",
        *[f"  - {n}" for n in notes],
        "",
        "Свод по курсам (course_summary.csv):",
        summary.to_string(index=False),
    ]
    text = "\n".join(lines)
    STATS_FILE.write_text(text, encoding="utf-8")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-courses", type=int, default=DEFAULT_MAX_COURSES)
    ap.add_argument("--courses", help="список ID через запятую, например CS-6200,CS-6515")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--base", default=BASE_URL, help="базовый URL (для тестов)")
    if "ipykernel" in sys.modules:
        args = ap.parse_args([])
    else:
        args = ap.parse_args()

    setup_logging()
    log.info("=== Старт сбора: %s, лимит курсов: %d ===", args.base, args.max_courses)
    t0 = time.time()
    courses = [c.strip().upper() for c in args.courses.split(",")] if args.courses else None
    rows, failed = scrape(args.base, courses, args.max_courses, args.fresh)
    df, notes = build_dataset(rows)
    if df.empty:
        log.error("Датасет пуст — проверьте лог и сеть")
        return
    write_outputs(df, notes, failed)
    log.info("=== Готово: %d записей за %.0f с ===", len(df), time.time() - t0)


if __name__ == "__main__":
    main()