"""
ПЗ5. Сбор связей между акторами из публичных issue/PR GitHub.

Переиспользован метод ПЗ3: GitHub GraphQL API, PAT из .env, requests.Session + Retry,
курсорная пагинация (pageInfo.endCursor), проактивный контроль rateLimit, чекпоинты.
Изменение относительно ПЗ3: вместо карточек репозиториев сохраняются СВЯЗИ
«источник -> приёмник, тип, timestamp».

Типы рёбер (направленные):
  reply   : автор комментария -> автор issue/PR (ответил автору обсуждения)
  mention : автор комментария -> @упомянутый пользователь
Текст комментариев НЕ сохраняется (минимизация данных). Для проверки однотипности
(этап 7) сохраняется только соль-хэш нормализованного текста: body_hash.

Запуск:
  python graph_collector.py --repo networkx/networkx --max-pages 20
Результат: edges_private.csv (реальные логины, НЕ сдавать), checkpoint_graph.jsonl
"""
import argparse, csv, hashlib, json, logging, os, re, time
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.github.com/graphql"
PAGE_SIZE = 25            # issue/PR на страницу
COMMENTS_PER_ITEM = 50    # комментариев на issue/PR (больше -> усечение, фиксируется в логе)
SOFT_PAUSE = 0.3
OUT_CSV = Path("edges_private.csv")
CHECKPOINT = Path("checkpoint_graph.jsonl")
HASH_SALT = "pz5-local-salt"   # хэш нужен только для сравнения внутри датасета

QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String, $cfirst: Int!) {
  rateLimit { limit cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    %s(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        createdAt
        author { login __typename }
        comments(first: $cfirst) {
          totalCount
          nodes { createdAt body author { login __typename } }
        }
      }
    }
  }
}
"""

MENTION_RE = re.compile(r"(?<![\w`])@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))")
log = logging.getLogger("pz5")


def build_session(token: str) -> requests.Session:
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["POST"], respect_retry_after_header=True, raise_on_status=False)
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json",
                      "User-Agent": "mirea-digital-footprint-practice-5"})
    return s


def graphql_request(session, query, variables):
    resp = session.post(API, json={"query": query, "variables": variables}, timeout=60)
    if resp.status_code != 200:
        log.error("Статус %s, тело: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        log.error("GraphQL errors: %s", payload["errors"])
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    rl = payload["data"].get("rateLimit") or {}
    limit, rem = rl.get("limit"), rl.get("remaining")
    log.info("Лимит: остаток %s из %s (cost %s)", rem, limit, rl.get("cost"))
    if limit and rem is not None and rem / limit < 0.05:
        log.warning("Остаток лимита < 5%% — пауза 5 с")
        time.sleep(5)
    return payload["data"]


def body_hash(body: str) -> str:
    norm = re.sub(r"\s+", " ", re.sub(r"@\w+|https?://\S+|\d+", "", (body or "").lower())).strip()
    if len(norm) < 3:
        return ""
    return hashlib.sha256((HASH_SALT + norm).encode()).hexdigest()[:12]


def actor(node):
    a = (node or {}).get("author")
    if not a or not a.get("login"):
        return None, None          # удалённый аккаунт (ghost) — пропускаем
    return a["login"], a.get("__typename", "User")


def extract_edges(item: dict, kind: str):
    """Из одного issue/PR -> список рёбер. Текст не возвращается."""
    edges = []
    owner, otype = actor(item)
    com = item["comments"]
    if com["totalCount"] > len(com["nodes"]):
        log.warning("%s #%s: усечено %d из %d комментариев", kind, item["number"],
                    len(com["nodes"]), com["totalCount"])
    for c in com["nodes"]:
        src, stype = actor(c)
        if not src:
            continue
        h = body_hash(c.get("body"))
        if owner and src != owner:
            edges.append(dict(source=src, target=owner, type="reply", timestamp=c["createdAt"],
                              thread=f"{kind}#{item['number']}", source_type=stype,
                              target_type=otype, body_hash=h))
        for m in set(MENTION_RE.findall(c.get("body") or "")):
            if m.lower() != src.lower():
                edges.append(dict(source=src, target=m, type="mention", timestamp=c["createdAt"],
                                  thread=f"{kind}#{item['number']}", source_type=stype,
                                  target_type="", body_hash=h))
    return edges


def collect(session, owner, name, kinds, max_pages):
    all_edges, errors = [], []
    for kind in kinds:                       # issues, pullRequests
        cursor, page = None, 0
        q = QUERY % kind
        while page < max_pages:
            vars_ = dict(owner=owner, name=name, first=PAGE_SIZE, after=cursor, cfirst=COMMENTS_PER_ITEM)
            try:
                data = graphql_request(session, q, vars_)
            except (requests.RequestException, RuntimeError) as exc:
                log.error("Остановка на %s, стр. %d: %s", kind, page + 1, exc)
                errors.append(str(exc)); break
            block = data["repository"][kind]
            nodes = block["nodes"]
            if not nodes:
                break
            page_edges = [e for it in nodes for e in extract_edges(it, kind)]
            all_edges.extend(page_edges)
            with CHECKPOINT.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": kind, "cursor": block["pageInfo"]["endCursor"],
                                    "edges": page_edges}, ensure_ascii=False) + "\n")
            page += 1
            log.info("%s: стр. %d, рёбер накоплено %d", kind, page, len(all_edges))
            if not block["pageInfo"]["hasNextPage"]:
                log.info("%s: hasNextPage=false", kind); break
            cursor = block["pageInfo"]["endCursor"]
            time.sleep(SOFT_PAUSE)
    return all_edges, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="networkx/networkx")
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--kinds", default="issues,pullRequests")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("graph_collect.log", encoding="utf-8")])
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        log.error("GITHUB_TOKEN не задан (.env)"); raise SystemExit(1)
    owner, name = args.repo.split("/")
    CHECKPOINT.unlink(missing_ok=True)
    t0 = time.perf_counter()
    edges, errors = collect(build_session(token), owner, name, args.kinds.split(","), args.max_pages)
    fields = ["source", "target", "type", "timestamp", "thread", "source_type", "target_type", "body_hash"]
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(edges)
    log.info("Готово: %d рёбер за %.1f с, ошибок: %d -> %s", len(edges), time.perf_counter() - t0,
             len(errors), OUT_CSV)


if __name__ == "__main__":
    main()
