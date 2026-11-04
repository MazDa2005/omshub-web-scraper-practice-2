"""
ПЗ5. Этапы 3-8: граф, центральности, сообщества, визуализация, аудит топ-3, анонимизация.
Вход : edges_private.csv (реальные логины)  Выход: папка out/ (всё анонимизировано,
кроме private/label_map.csv и edges_private.csv — их НЕ сдавать).
Запуск: python analyze_graph.py [--edges edges_private.csv] [--seed 42]
"""
import argparse, json, math, random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nxc

p = argparse.ArgumentParser()
p.add_argument("--edges", default="edges_private.csv")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--k-sample", type=int, default=500, help="k для приближённой betweenness, если n > k")
p.add_argument("--plot-top", type=int, default=250, help="сколько узлов рисовать")
args = p.parse_args()
OUT, PRIV = Path("out"), Path("out/private")
OUT.mkdir(exist_ok=True); PRIV.mkdir(parents=True, exist_ok=True)

# ---------- Этап 8 (раньше всех остальных): анонимизация ----------
df = pd.read_csv(args.edges, encoding="utf-8-sig").fillna("")
df["source"], df["target"] = df["source"].str.lower(), df["target"].str.lower()  # логины GitHub нечувствительны к регистру
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
users = sorted(set(df.source) | set(df.target))
rng = random.Random(args.seed); shuffled = users[:]; rng.shuffle(shuffled)   # метка не зависит от ранга
label = {u: f"user_{i+1:03d}" for i, u in enumerate(shuffled)}
pd.DataFrame({"real_id": list(label), "label": list(label.values())}).to_csv(PRIV / "label_map.csv", index=False)
anon = df.copy()
anon["source"], anon["target"] = df.source.map(label), df.target.map(label)
anon = anon.drop(columns=["thread"]).assign(thread_id=df.groupby("thread").ngroup().map(lambda i: f"t_{i+1:04d}"))
anon.to_csv(OUT / "edges_anon.csv", index=False)
types = {}   # label -> User/Bot по данным API
for s, t in zip(df.source, df.source_type): types[label[s]] = t
for s, t in zip(df.target, df.target_type):
    if t: types.setdefault(label[s], t)
df = anon  # дальше работаем только с анонимными метками

# ---------- Этап 3: граф ----------
agg = (df.groupby(["source", "target"]).agg(weight=("type", "size"), first=("timestamp", "min"),
       last=("timestamp", "max")).reset_index())
G = nx.DiGraph()
for r in agg.itertuples(): G.add_edge(r.source, r.target, weight=r.weight)
U = G.to_undirected()
lcc_nodes = max(nx.connected_components(U), key=len)
stats = dict(type="DiGraph (направленный, взвешенный: weight = число взаимодействий)",
             interactions=len(df), nodes=G.number_of_nodes(), edges=G.number_of_edges(),
             density=round(nx.density(G), 5), weak_components=nx.number_weakly_connected_components(G),
             strong_components=nx.number_strongly_connected_components(G),
             lcc_share=round(len(lcc_nodes) / G.number_of_nodes(), 3),
             reciprocity=round(nx.reciprocity(G), 3), edge_types=df.type.value_counts().to_dict(),
             bots_in_graph=sum(1 for v in types.values() if v == "Bot"))

# ---------- Этап 4: центральности (разные семейства) ----------
n = G.number_of_nodes()
k = args.k_sample if n > args.k_sample else None
stats["betweenness_mode"] = f"приближённая, k={k}, seed={args.seed}" if k else "точная"
deg_c = nx.degree_centrality(G)
indeg = dict(G.in_degree()); outdeg = dict(G.out_degree())
btw = nx.betweenness_centrality(G, k=k, seed=args.seed, normalized=True)
clo = nx.closeness_centrality(U.subgraph(lcc_nodes), wf_improved=True)   # только по наибольшей компоненте
pr = nx.pagerank(G, weight="weight")
M = pd.DataFrame({"degree_c": deg_c, "in_degree": indeg, "out_degree": outdeg,
                  "betweenness": btw, "closeness_lcc": pd.Series(clo), "pagerank": pr})
M["actor_type"] = pd.Series(types)
M.to_csv(OUT / "metrics_all_nodes.csv", index_label="node")
top = {}
for m in ["degree_c", "betweenness", "closeness_lcc", "pagerank"]:
    t = M[m].sort_values(ascending=False).head(10).round(5).rename_axis("node").reset_index()
    t.to_csv(OUT / f"top10_{m}.csv", index=False); top[m] = t

# ---------- Этап 5: сообщества ----------
W = nx.Graph()
for u, v, d in G.edges(data=True):
    w = d["weight"] + (G[v][u]["weight"] if G.has_edge(v, u) else 0)   # симметризация: сумма весов обоих направлений
    W.add_edge(u, v, weight=w)
louv = nxc.louvain_communities(W, weight="weight", seed=args.seed)
greedy = nxc.greedy_modularity_communities(W, weight="weight")
Q_l, Q_g = nxc.modularity(W, louv, weight="weight"), nxc.modularity(W, greedy, weight="weight")
use, used_name, Qv = (louv, "Louvain (networkx.louvain_communities)", Q_l) if Q_l >= Q_g else (greedy, "greedy modularity", Q_g)
use = sorted(use, key=len, reverse=True)
comm = {u: i for i, c in enumerate(use) for u in c}
M["community"] = pd.Series(comm); M.to_csv(OUT / "metrics_all_nodes.csv", index_label="node")
csz = pd.DataFrame({"community": range(len(use)), "size": [len(c) for c in use]})
csz["share"] = (csz["size"] / n).round(3); csz.to_csv(OUT / "communities.csv", index=False)
comm_info = dict(method=used_name, count=len(use), Q=round(Qv, 4), Q_louvain=round(Q_l, 4),
                 Q_greedy=round(Q_g, 4), singletons=int((csz["size"] == 1).sum()),
                 sizes_top5=csz["size"].head(5).tolist())

# ---------- Этап 6: визуализация ----------
core_nodes = M.loc[M.index.isin(lcc_nodes)].eval("betweenness + degree_c").sort_values(ascending=False).head(args.plot_top).index  # только наибольшая компонента
H = W.subgraph(core_nodes)
pos = nx.spring_layout(H, seed=args.seed, k=3.0 / math.sqrt(max(H.number_of_nodes(), 1)), iterations=200, weight="weight")
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(13, 10))
nx.draw_networkx_edges(H, pos, alpha=0.15, width=0.5, ax=ax)
sizes = [30 + 3500 * float(M.at[v, "betweenness"]) ** 0.7 + 400 * float(M.at[v, "degree_c"]) for v in H]
nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=[cmap(comm[v] % 20) for v in H],
                       linewidths=0.4, edgecolors="white", ax=ax)
top_lab = {v: v for v in M["betweenness"].sort_values(ascending=False).head(6).index if v in H}
nx.draw_networkx_labels(H, pos, labels=top_lab, font_size=9, font_weight="bold", ax=ax,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
ax.set_title(f"Граф взаимодействий (анонимизирован): {H.number_of_nodes()} из {n} узлов; "
             f"размер — betweenness+degree, цвет — сообщество ({used_name}, Q={Qv:.3f})", fontsize=10)
ax.axis("off"); fig.tight_layout(); fig.savefig(OUT / "graph.png", dpi=170); plt.close(fig)
nx.write_gexf(G, OUT / "graph_anon.gexf")

# ---------- Этап 7: хаб или бот-артефакт (топ-3 по любой метрике) ----------
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
df["t"] = (df["timestamp"] - EPOCH).dt.total_seconds()

def burstiness(t):
    """Goh–Barabási B по интервалам между событиями: -1 регулярно, 0 пуассон, +1 бёрсты."""
    d = np.diff(np.sort(t))
    if len(d) < 3 or (d.std() + d.mean()) == 0: return float("nan")
    return float((d.std() - d.mean()) / (d.std() + d.mean()))

def max_window_share(t, hours=1):
    s = np.sort(t); w = hours * 3600; j = best = 0
    for i in range(len(s)):
        while s[i] - s[j] > w: j += 1
        best = max(best, i - j + 1)
    return best / len(s)

cand = []
for m in ["degree_c", "betweenness", "closeness_lcc", "pagerank"]:
    cand += list(M[m].sort_values(ascending=False).head(3).index)
cand = list(dict.fromkeys(cand))
rows = []
for v in cand:
    ev = df[df.source == v]; t = ev["t"].values
    out_w, in_w = len(ev), int((df.target == v).sum())
    nbrs = set(W.neighbors(v))
    recip = float(np.mean([G.has_edge(u, v) and G.has_edge(v, u) for u in nbrs])) if nbrs else 0.0
    h = ev["body_hash"][ev["body_hash"] != ""]
    rep = float(h.value_counts().iloc[0] / len(h)) if len(h) else float("nan")
    cl = nx.clustering(W, v)
    win = max_window_share(t) if out_w >= 5 else float("nan")
    flags = []
    if types.get(v) == "Bot": flags.append("тип аккаунта Bot (по API)")
    if out_w > 20 and in_w / out_w < 0.05: flags.append("почти нет входящих")
    if win == win and win > 0.5: flags.append(">50% действий в окне 1 ч")
    if len(nbrs) > 20 and cl < 0.02: flags.append("звёздная окрестность (clustering<0.02)")
    if rep == rep and rep > 0.3 and len(h) >= 10: flags.append("повтор шаблона >30%")
    verdict = ("вероятный бот-артефакт" if (types.get(v) == "Bot" or len(flags) >= 3) else
               "подлинный хаб" if not flags else "недостаточно данных (проверить вручную)")
    rows.append(dict(node=v, actor_type=types.get(v, ""), neighbors=len(nbrs), out_w=out_w, in_w=in_w,
                     in_out_ratio=round(in_w / out_w, 3) if out_w else None, reciprocated_nbrs=round(recip, 3),
                     clustering=round(cl, 4), burstiness_B=round(burstiness(t), 3),
                     max_1h_window_share=None if win != win else round(win, 3),
                     top_template_share=None if rep != rep else round(rep, 3),
                     communities_touched=len({comm[u] for u in nbrs}), own_community=comm[v],
                     flags="; ".join(flags), verdict=verdict))
audit = pd.DataFrame(rows); audit.to_csv(OUT / "top_nodes_audit.csv", index=False)

# ---------- Приватность: насколько узел уникален по окрестности ----------
sig = {v: (W.degree(v), tuple(sorted(W.degree(u) for u in W.neighbors(v)))) for v in W}
sig_counts = pd.Series(list(sig.values())).value_counts()
unique_share = float(np.mean([sig_counts[sig[v]] == 1 for v in W]))
deg_seq = pd.Series(dict(W.degree()))
privacy = dict(leaf_share=round(float((deg_seq == 1).mean()), 3),
               deg_le2_share=round(float((deg_seq <= 2).mean()), 3),
               unique_1hop_signature_share=round(unique_share, 3),
               threads=int(df.thread_id.nunique()))

# ---------- Сводка для отчёта ----------
summary = dict(graph=stats, communities=comm_info, privacy=privacy)
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
for m, t in top.items(): print(f"\nTOP-10 {m}\n", t.to_string(index=False))
print("\nАУДИТ ТОП-УЗЛОВ\n", audit[["node", "actor_type", "in_out_ratio", "clustering", "max_1h_window_share",
                                   "communities_touched", "flags", "verdict"]].to_string(index=False))
