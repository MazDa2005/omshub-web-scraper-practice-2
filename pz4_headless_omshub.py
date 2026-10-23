#!/usr/bin/env python3
"""
Практическое занятие 4. Сбор данных с динамической страницы: headless-браузеры
и перехват сетевых запросов.
Дисциплина: «Сбор первичных данных цифрового следа».

Цель: https://omshub.org/course/<ID> — список отзывов на странице курса.
Продолжение кейса ПЗ2/ПЗ3 (Georgia Tech / OMSCS).

Почему headless, а не requests (этап 1, см. также ПЗ2 разд. 2 «Схема пагинации»):
статический HTML страницы курса содержит только 20 последних отзывов
(«20 of 163 reviews (load more below)»), остальные подгружаются кнопкой
«Load more reviews» через AJAX-запрос к /api/reviews/recent — этот путь запрещён
в robots.txt сайта (Disallow: /api/), поэтому прямой requests к нему не выполняется
(этап 5), а используется браузерная автоматизация, нажимающая ту же кнопку, что и
обычный пользователь.

Запуск:
    pip install playwright pandas requests
    playwright install chromium
    python pz4_headless_omshub.py --course CS-6310

"""

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser

import pandas as pd
import requests
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE = "https://omshub.org"
MIN_RECORDS = 150
MAX_ROUNDS = 60          # предел итераций клика (этап 4: обязательное условие остановки)
STABLE_LIMIT = 3         # сколько раундов подряд без прироста считаем завершением
TIMEOUT = 15

# Единый реалистичный User-Agent для ВСЕХ запросов (requests и Playwright) — важно для
# согласованности: сайт может по-разному отвечать клиентам без узнаваемого UA (см. историю правок).
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"}

CARD_SEL = 'div[data-review-card="true"]'
LOAD_MORE_SEL = 'button:has-text("Load more reviews")'
BADGE_SEL = "span.mantine-Badge-label"
COURSE_LINE_SEL = 'p[data-size="md"]'
SMALL_SEL = 'p[data-size="xs"]'
STAT_SEL = 'p[data-size="lg"]'
BODY_SEL = "div.review-body"

OUT_DIR = Path(".")
LOG_FILE = OUT_DIR / "pz4_headless.log"
CSV_FILE = OUT_DIR / "dynamic_page_dataset.csv"
STATS_FILE = OUT_DIR / "pz4_stats.txt"

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)                    # React/Next.js hydration-комментарии
TOTAL_RE = re.compile(r"(\d+)\s+of\s+(\d+)\s+review", re.I)     # применяется ПОСЛЕ очистки от комментариев
SEMESTER_RE = re.compile(r"(SPRING|SUMMER|FALL)\s+(\d{4})", re.I)
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
COURSE_RE = re.compile(r"^\s*([A-Z]+-\d+)\s*:\s*(.+?)\s*$")

log = logging.getLogger("pz4")


def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    for h in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


# --------------------------------------------------------------------------- этап 1

def step1_confirm_dynamic(url: str) -> tuple[int, int | None]:
    """Контрольный GET без браузера: подтверждаем, что requests не отдаёт все карточки."""
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    raw = resp.text
    count_in_html = raw.count('data-review-card="true"')

    cleaned = COMMENT_RE.sub("", raw)   # см. «История правок» (1)
    total_m = TOTAL_RE.search(cleaned)
    total = int(total_m.group(2)) if total_m else None

    log.info("Контрольный запрос requests.get(%s): HTTP %s, карточек в сыром HTML: %d, "
              "заявлено всего отзывов на сайте: %s", url, resp.status_code, count_in_html, total)
    if total is None:
        log.warning("Не удалось извлечь общее число отзывов из текста страницы (разметка могла "
                     "измениться) — сравнение «N из total» будет неполным, но сбор продолжится")
    else:
        assert count_in_html < total, \
            "В статическом HTML уже есть все записи — headless-подход не требуется"
    return count_in_html, total


# --------------------------------------------------------------------------- парсинг карточек

def first_int(text: str | None):
    m = re.search(r"\d+", text or "")
    return int(m.group()) if m else None


def parse_cards(page) -> list[dict]:
    """Извлекает структурные поля карточек. Текст отзыва и ID автора НЕ собираются (см. ПЗ2/ПЗ3:
    в тексте отзывов встречаются имена преподавателей/ассистентов — потенциальные ПДн)."""
    cards = page.query_selector_all(CARD_SEL)
    rows = []
    for card in cards:
        labels = [s.inner_text().strip() for s in card.query_selector_all(BADGE_SEL)]
        sem = SEMESTER_RE.search(labels[0]) if labels else None

        course_el = card.query_selector(COURSE_LINE_SEL)
        cm = COURSE_RE.match(course_el.inner_text()) if course_el else None

        review_date = None
        for p in card.query_selector_all(SMALL_SEL):
            t = p.inner_text().strip()
            if DATE_RE.match(t):
                review_date = datetime.strptime(t, "%m/%d/%Y").date().isoformat()
                break

        stats = [first_int(p.inner_text()) for p in card.query_selector_all(STAT_SEL)][:3]
        stats += [None] * (3 - len(stats))

        body = card.query_selector(BODY_SEL)
        rows.append({
            "course_id": cm.group(1) if cm else None,
            "course_name": cm.group(2) if cm else None,
            "semester": f"{sem.group(1).upper()} {sem.group(2)}" if sem else None,
            "review_date": review_date,
            "workload_hrs": stats[0],
            "difficulty": stats[1],
            "overall": stats[2],
            "verified_gt": "Verified" in labels[1:],
            "legacy": "Legacy" in labels[1:],
            "body_words": len(body.inner_text().split()) if body else None,  # только длина, не текст
        })
    return rows


# --------------------------------------------------------------------------- этапы 2-4

def step2_4_load_all(page, url: str) -> tuple[list[dict], dict]:
    """Открывает страницу headless-браузером и кликает «Load more reviews» до стабилизации."""
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector(CARD_SEL, state="visible", timeout=15_000)   # явное ожидание, не sleep()
    log.info("Страница открыта headless-браузером, заголовок: %s", page.title())

    stable_rounds = 0
    current = page.eval_on_selector_all(CARD_SEL, "els => els.length")
    log.info("После первой загрузки (DOM): карточек = %d", current)

    round_ = 0
    stop_reason = "max_rounds_reached"
    for round_ in range(1, MAX_ROUNDS + 1):
        btn = page.query_selector(LOAD_MORE_SEL)
        if btn is None or not btn.is_visible():
            stop_reason = "button_gone"
            log.info("Раунд %d: кнопка «Load more reviews» не найдена/не видна — подгрузка завершена "
                      "(карточек: %d)", round_, current)
            break

        btn.scroll_into_view_if_needed()
        btn.click()
        try:
            # Явное условие ожидания прироста DOM вместо time.sleep() (требование этапа 3-4)
            page.wait_for_function(
                "prev => document.querySelectorAll('div[data-review-card=\"true\"]').length > prev",
                arg=current, timeout=6_000,
            )
        except PWTimeout:
            log.warning("Раунд %d: число карточек не выросло за 6 с после клика", round_)

        new_count = page.eval_on_selector_all(CARD_SEL, "els => els.length")
        log.info("Раунд %d: карточек стало %d (было %d)", round_, new_count, current)

        if new_count == current:
            stable_rounds += 1
            if stable_rounds >= STABLE_LIMIT:
                stop_reason = "stabilized"
                log.info("%d раунда подряд без прироста — останавливаемся (условие стабилизации)",
                          stable_rounds)
                break
        else:
            stable_rounds = 0
        current = new_count
        time.sleep(0.3)  # мягкая пауза между кликами (вежливость к серверу, не замена явного ожидания)
    else:
        log.warning("Достигнут предел итераций (%d) без подтверждённой стабилизации — "
                     "фиксируем как причину остановки", MAX_ROUNDS)

    rows = parse_cards(page)
    log.info("Итог подгрузки: %d карточек, причина остановки цикла: %s, раундов: %d",
              len(rows), stop_reason, round_)
    return rows, {"stop_reason": stop_reason, "rounds": round_}


# --------------------------------------------------------------------------- этап 5

def step5_try_api(course_id: str) -> dict:
    """Этап 5 (обязательный): находим эндпоинт, но НЕ обходим запрет robots.txt.

    Использует urllib.robotparser (как в ПЗ2) вместо проверки подстроки "/api/" in text —
    самодельная проверка один раз уже дала ложный отрицательный результат, когда запрос
    без заголовков получил нетиповой ответ (см. «История правок» (2)).
    """
    api_url = f"{BASE}/api/reviews/recent?limit=20&offset=20"  # обнаружен во вкладке Network, см. ПЗ2
    info = {"api_url": api_url, "robots_disallowed": None, "attempted": False, "note": ""}

    rp = robotparser.RobotFileParser()
    try:
        resp = requests.get(f"{BASE}/robots.txt", headers=HEADERS, timeout=TIMEOUT)
        log.info("GET %s/robots.txt -> HTTP %s, длина ответа: %d байт", BASE, resp.status_code, len(resp.text))
        log.info("Первые строки robots.txt:\n%s", "\n".join(resp.text.splitlines()[:6]))
        rp.parse(resp.text.splitlines() if resp.status_code == 200 else [])
    except requests.RequestException as exc:
        info["note"] = f"robots.txt не получен: {exc}"
        log.warning(info["note"])
        return info

    allowed = rp.can_fetch(UA, api_url)
    info["robots_disallowed"] = not allowed
    log.info("Обнаруженный во вкладке Network JSON-эндпоинт: %s", api_url)
    log.info("robotparser.can_fetch(%r, %s) = %s", UA, api_url, allowed)

    return info


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default="CS-6310", help="код курса на omshub.org, напр. CS-6310")
    args = ap.parse_args()

    setup_logging()
    url = f"{BASE}/course/{args.course}"
    log.info("=== ПЗ4: старт, цель=%s ===", url)
    t0 = time.perf_counter()

    count_static, total_declared = step1_confirm_dynamic(url)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=UA,
            viewport={"width": 1366, "height": 900},
        )
        page = context.new_page()
        rows, load_info = step2_4_load_all(page, url)
        browser.close()

    elapsed = time.perf_counter() - t0
    api_info = step5_try_api(args.course)

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")

    # Гипотетическая оценка прямого HTTP-подхода: N запросов по 20 записей, без запуска браузера.
    # Чисто иллюстративная величина для сравнения порядков — не выполняется физически (этично запрещено).
    hyp_requests = max(1, (len(df) + 19) // 20)
    hyp_seconds = hyp_requests * 0.3  # условные 0.3 с на лёгкий JSON GET, для сравнения порядка величины

    stats = [
        f"Источник: {url}",
        f"Дата и время сбора (UTC): {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')}",
        f"Карточек в статическом HTML (requests, этап 1): {count_static} из заявленных {total_declared}",
        f"Карточек после headless-подгрузки: {len(df)}",
        f"Причина остановки цикла подгрузки: {load_info['stop_reason']}, раундов кликов: {load_info['rounds']}",
        f"Соответствие минимуму в 150 записей: {'да' if len(df) >= MIN_RECORDS else 'НЕТ — ' + str(len(df))}",
        "",
        f"Время headless-сценария: {elapsed:.1f} с ({elapsed/max(len(df),1):.3f} с/запись)",
        f"Гипотетическая оценка прямого API-подхода (не выполнялась физически — см. этап 5): "
        f"~{hyp_requests} запросов, ~{hyp_seconds:.1f} с без учёта запуска браузера",
        "",
        f"Найденный JSON-эндпоинт: {api_info['api_url']}",
        f"robots.txt запрещает /api/: {api_info['robots_disallowed']}",
        f"Прямой запрос к эндпоинту выполнялся: {api_info['attempted']} ({api_info['note']})",
        "",
        "Пропуски по полям:",
        df.isna().sum().to_string() if not df.empty else "(датасет пуст)",
    ]
    text = "\n".join(stats)
    STATS_FILE.write_text(text, encoding="utf-8")
    print("\n" + text)
    log.info("=== Готово: %d записей за %.1f с ===", len(df), elapsed)


if __name__ == "__main__":
    main()