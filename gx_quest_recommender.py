
# gx_recommender.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecommendConfig:
    k_total: int = 5              # 4 + 1 exploration
    k_exploit: int = 4
    k_explore: int = 1
    mmr_lambda: float = 0.70      # 1.0=類似度重視, 0.0=多様性重視
    random_state: int = 42
    # score weights
    w_sim: float = 0.60
    w_succ: float = 0.20
    w_imp: float = 0.20


def _normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.all(np.isfinite(x)) and x.size > 0:
        mn, mx = np.min(x), np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)


def compute_success_prob(
    quests: pd.DataFrame,
    user_history: pd.DataFrame,
    default: float = 0.5
) -> pd.Series:
    """
    success_prob をミッションごとに推定。
    user_history: columns = [quest_id, achieved] (+ optional timestamp)
    - 全体達成率
    - 同カテゴリ達成率
    を平均して作る（冷スタは default）。
    """
    if user_history is None or user_history.empty:
        return pd.Series(default, index=quests.index)

    hist = user_history.copy()
    hist = hist[hist["quest_id"].isin(quests["quest_id"])]
    if hist.empty:
        return pd.Series(default, index=quests.index)

    # overall achieved rate
    overall = float(hist["achieved"].mean()) if "achieved" in hist.columns else default

    # category achieved rate
    mcat = quests[["quest_id", "category"]]
    hist2 = hist.merge(mcat, on="quest_id", how="left")
    cat_rate = (
        hist2.groupby("category")["achieved"].mean()
        if "achieved" in hist2.columns else pd.Series(dtype=float)
    )

    probs = []
    for _, row in quests.iterrows():
        c = row["category"]
        cr = float(cat_rate.get(c, overall)) if len(cat_rate) else overall
        probs.append(0.5 * overall + 0.5 * cr)

    return pd.Series(probs, index=quests.index).clip(0.0, 1.0)


def build_text_similarity(
    quests: pd.DataFrame,
    achieved_titles: Sequence[str],
    candidate_titles: Sequence[str],
) -> np.ndarray:
    """
    TF-IDF で類似度（cosine）を返す。
    achieved_titles の平均ベクトル vs candidate_titles の類似度。
    """
    corpus = list(quests["title"].astype(str).values)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1
    )
    X = vectorizer.fit_transform(corpus)

    # map title -> vector row index
    title_to_idx = {t: i for i, t in enumerate(quests["title"].astype(str).values)}

    ach_idx = [title_to_idx[t] for t in achieved_titles if t in title_to_idx]
    cand_idx = [title_to_idx[t] for t in candidate_titles if t in title_to_idx]

    if len(cand_idx) == 0:
        return np.array([])

    if len(ach_idx) == 0:
        # cold start: similarity all zeros
        return np.zeros(len(cand_idx), dtype=float)

    ach_vec = X[ach_idx].mean(axis=0)
    # scipy sparse mean may return numpy.matrix; convert to ndarray for sklearn
    ach_vec = np.asarray(ach_vec)
    cand_vec = X[cand_idx]

    sims = cosine_similarity(ach_vec, cand_vec).ravel()
    return np.asarray(sims, dtype=float)


def mmr_rerank(
    base_scores: np.ndarray,
    sim_to_profile: np.ndarray,
    pairwise_sim: np.ndarray,
    k: int,
    lam: float
) -> List[int]:
    """
    MMR: argmax_i lam*sim(profile,i) + (1-lam)*base_score(i) - (1-lam)*max_{j in S} sim(i,j)
    ※ここでは base_score も活用して、類似度偏重を緩和。
    Returns: selected indices in candidate list order.
    """
    n = len(base_scores)
    if n == 0 or k <= 0:
        return []

    selected: List[int] = []
    remaining = list(range(n))

    # first pick: highest combined of profile-sim and base
    first = int(np.argmax(lam * sim_to_profile + (1 - lam) * base_scores))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < k:
        mmr_vals = []
        for i in remaining:
            diversity_penalty = max(pairwise_sim[i, j] for j in selected) if selected else 0.0
            val = lam * sim_to_profile[i] + (1 - lam) * base_scores[i] - (1 - lam) * diversity_penalty
            mmr_vals.append((val, i))
        mmr_vals.sort(reverse=True, key=lambda x: x[0])
        best_i = mmr_vals[0][1]
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


def recommend_quests(
    quests: pd.DataFrame,
    user_history: Optional[pd.DataFrame] = None,
    *,
    context_location: Optional[str] = None,
    allow_repeatable_only: bool = True,
    exclude_completed: bool = True,
    config: RecommendConfig = RecommendConfig(),
) -> pd.DataFrame:
    """
    ご提示ロジックに沿う推薦（MMR + 探索枠）。
    quests: columns must include [quest_id,title,category,difficulty,impact,location,repeatable]
    user_history: columns [quest_id, achieved] (+ optional columns)
    context_location:
        - None: location filterなし
        - "家" / "外" / "職場/学校" / "どこでも" などを渡す
    """

    rng = np.random.default_rng(config.random_state)

    m = quests.copy()

    # 1) 候補集合の作成
    if allow_repeatable_only and "repeatable" in m.columns:
        m = m[m["repeatable"] == "はい"].copy()

    if context_location is not None and "location" in m.columns:
        # "どこでも" は常に許可、"家/外" は部分一致
        def loc_ok(loc: str) -> bool:
            loc = str(loc)
            if loc == "どこでも":
                return True
            if context_location in loc.split("/"):
                return True
            return loc == context_location

        m = m[m["location"].apply(loc_ok)].copy()

    achieved_ids = set()
    achieved_titles: List[str] = []
    if user_history is not None and not user_history.empty and "quest_id" in user_history.columns:
        achieved_rows = user_history[user_history.get("achieved", True) == True]
        achieved_ids = set(achieved_rows["quest_id"].tolist())
        # titles for similarity profile
        achieved_titles = quests[quests["quest_id"].isin(achieved_ids)]["title"].astype(str).tolist()

    if exclude_completed and achieved_ids:
        m = m[~m["quest_id"].isin(achieved_ids)].copy()

    if m.empty:
        return m

    # 2) スコアリング
    # similarity: achieved profile vs each candidate
    cand_titles = m["title"].astype(str).tolist()
    sim = build_text_similarity(quests, achieved_titles, cand_titles)  # length = len(candidates)
    if sim.size == 0:
        sim = np.zeros(len(m), dtype=float)

    sim_n = _normalize_0_1(sim)

    # success_prob
    succ = compute_success_prob(m, user_history)
    succ_n = _normalize_0_1(succ.values)

    # impact (1-5) -> normalize
    imp = m["impact"].astype(float).values if "impact" in m.columns else np.ones(len(m))
    imp_n = _normalize_0_1(imp)

    base = config.w_sim * sim_n + config.w_succ * succ_n + config.w_imp * imp_n

    # pairwise similarity for MMR (candidate-candidate)
    # Use TF-IDF on candidate titles only for speed
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    Xc = vectorizer.fit_transform(cand_titles)
    pairwise = cosine_similarity(Xc, Xc)

    # 3) 多様性でリランキング (MMR)
    k_exploit = min(config.k_exploit, len(m))
    selected_idx = mmr_rerank(base, sim_n, pairwise, k_exploit, config.mmr_lambda)

    exploit = m.iloc[selected_idx].copy()
    exploit = exploit.assign(
        similarity=sim_n[selected_idx],
        success_prob=succ.values[selected_idx],
        score=base[selected_idx],
        reason="exploit"
    )

    # 4) 探索枠を混ぜる（上位4つ＋探索1つ）
    explore = pd.DataFrame(columns=exploit.columns)
    if config.k_explore > 0 and len(m) > len(exploit):
        remaining = m.drop(m.index[selected_idx]).copy()

        # 探索は「低〜中類似度」「未露出カテゴリ」「難易度は3前後」を優先
        # まずカテゴリ露出を計算
        hist = user_history if user_history is not None else pd.DataFrame()
        cat_counts = {}
        if not hist.empty and "quest_id" in hist.columns:
            tmp = hist.merge(quests[["quest_id", "category"]], on="quest_id", how="left")
            cat_counts = tmp["category"].value_counts().to_dict()

        rem_titles = remaining["title"].astype(str).tolist()
        rem_sim = build_text_similarity(quests, achieved_titles, rem_titles)
        rem_sim_n = _normalize_0_1(rem_sim) if rem_sim.size else np.zeros(len(remaining))

        # exploration weight: prefer lower similarity (novel), lower exposure, moderate difficulty, decent impact
        exposure = np.array([cat_counts.get(c, 0) for c in remaining["category"].tolist()], dtype=float)
        exposure_n = _normalize_0_1(exposure)
        diff = remaining["difficulty"].astype(float).values if "difficulty" in remaining.columns else np.ones(len(remaining))*3
        diff_pref = -np.abs(diff - 3.0)  # closer to 3 is better
        diff_n = _normalize_0_1(diff_pref)
        imp2 = remaining["impact"].astype(float).values if "impact" in remaining.columns else np.ones(len(remaining))*3
        imp2_n = _normalize_0_1(imp2)

        explore_score = (1 - rem_sim_n) * 0.45 + (1 - exposure_n) * 0.25 + diff_n * 0.15 + imp2_n * 0.15

        # sample top candidates then pick one deterministically by max (or weighted random)
        j = int(np.argmax(explore_score))
        pick = remaining.iloc[[j]].copy()

        # add columns to match exploit
        pick = pick.assign(
            similarity=float(rem_sim_n[j]),
            success_prob=float(compute_success_prob(pick, user_history).iloc[0]),
            score=float(explore_score[j]),
            reason="explore"
        )
        explore = pick

    out = pd.concat([exploit, explore], ignore_index=True)

    # tidy
    cols_front = ["quest_id", "title", "category", "difficulty", "cost", "location", "impact", "repeatable",
                 "similarity", "success_prob", "score", "reason"]
    cols = [c for c in cols_front if c in out.columns] + [c for c in out.columns if c not in cols_front]
    return out[cols].sort_values(by=["reason","score"], ascending=[True, False]).reset_index(drop=True)


if __name__ == "__main__":
    # quick demo (cold start)
    quests = pd.read_csv("gx_missions_with_metadata_5scale.csv", encoding="utf-8-sig")
    rec = recommend_quests(quests, user_history=None, context_location="家")
    print(rec.head(5).to_string(index=False))
