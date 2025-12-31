from datetime import datetime

import pandas as pd
import streamlit as st

# ---- Configuration ----
CSV_PATH = "gx_quests.csv"  # place the CSV next to this app
# If you keep the recommender in the same folder, import like this:
from gx_quest_recommender import RecommendConfig, recommend_quests


def load_quests(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {
        "quest_id",
        "title",
        "category",
        "difficulty",
        "cost",
        "location",
        "impact",
        "repeatable",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def init_state():
    if "achieved_ids" not in st.session_state:
        st.session_state.achieved_ids = set()
    if "history" not in st.session_state:
        # history rows: quest_id, achieved, timestamp
        st.session_state.history = []


def mark_achieved(quest_id: int):
    if quest_id in st.session_state.achieved_ids:
        return
    st.session_state.achieved_ids.add(int(quest_id))
    st.session_state.history.append(
        {
            "quest_id": int(quest_id),
            "achieved": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    # Update UI immediately (recompute recommendations and list)
    st.rerun()


def build_user_history_df() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(columns=["quest_id", "achieved", "timestamp"])
    return pd.DataFrame(st.session_state.history)


def _stars_html(value: int, max_value: int = 5) -> str:
    """Return HTML string of star rating."""
    value = int(max(0, min(max_value, value)))
    filled = "★" * value
    empty = "☆" * (max_value - value)
    # Gold for filled, light gray for empty
    return f"<span style='font-size: 18px; line-height: 1;'><span style='color:#f5c542'>{filled}</span><span style='color:#c7c7c7'>{empty}</span></span>"


def _category_badge(category: str):
    """Color-coded category badge using st.badge."""
    cat = str(category)
    color_map = {
        "電気・エネルギー": "orange",
        "移動・交通": "blue",
        "食・買い物": "green",
        "資源循環": "violet",
        "水・自然・学び": "gray",
    }
    color = color_map.get(cat, "gray")
    # Streamlit badge renders inline; keep short.
    st.badge(cat, color=color)


def quest_card(row: pd.Series, achieved: bool, key_prefix: str):
    with st.container(border=True):
        header = st.columns([7, 3])
        with header[0]:
            st.markdown(f"**{row['title']}**")
            # Category badge (colored)
            _category_badge(row["category"])

            # Stars for difficulty / impact
            diff = int(row.get("difficulty", 0))
            imp = int(row.get("impact", 0))
            st.markdown(
                f"難易度: {_stars_html(diff)}　　インパクト: {_stars_html(imp)}",
                unsafe_allow_html=True,
            )
        with header[1]:
            st.button(
                "達成 ✅" if not achieved else "達成済み",
                key=f"{key_prefix}_btn_{int(row['quest_id'])}",
                disabled=achieved,
                use_container_width=True,
                on_click=mark_achieved,
                args=(int(row["quest_id"]),),
            )


def main():
    st.set_page_config(
        page_title="GX with TQ レコメンド機能", initial_sidebar_state="collapsed"
    )
    init_state()

    st.title("GX with TQ レコメンド機能")

    cfg = RecommendConfig(k_total=5, k_exploit=4, k_explore=1, mmr_lambda=0.7)

    # Sidebar
    with st.sidebar:
        if st.button("達成状況をリセット"):
            st.session_state.achieved_ids = set()
            st.session_state.history = []
            st.success("リセットしました。")

        hist_df = build_user_history_df()
        if not hist_df.empty:
            st.download_button(
                "達成履歴CSVをダウンロード",
                data=hist_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="quest_history.csv",
                mime="text/csv",
            )

    # Load quests
    try:
        df_quests = load_quests(CSV_PATH)
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        st.info("CSV_PATH を確認し、CSVとこのアプリを同じフォルダに置いてください。")
        st.stop()

    # Build history
    user_history = build_user_history_df()

    # --- Recommendations ---
    st.subheader("おすすめクエスト")
    rec = recommend_quests(
        df_quests,
        user_history=user_history,
        context_location=None,  # ← 要件どおり None 固定
        allow_repeatable_only=True,
        exclude_completed=True,
        config=cfg,
    )

    if rec.empty:
        st.info(
            "おすすめできるクエストがありません（全て達成済み、または候補が絞り込まれました）。"
        )
    else:
        # cols = st.columns(5)
        for i, (_, row) in enumerate(rec.iterrows()):
            # with cols[i % 5]:
            achieved = int(row["quest_id"]) in st.session_state.achieved_ids
            quest_card(row, achieved, key_prefix="rec")

    st.divider()

    # --- Quest List ---
    st.subheader("クエスト一覧")

    achieved_mask = (
        df_quests["quest_id"].astype(int).isin(st.session_state.achieved_ids)
    )

    # Sort: unachieved first, achieved last. Within each: higher impact first, then lower difficulty
    df_unach = df_quests.loc[~achieved_mask].copy()
    df_ach = df_quests.loc[achieved_mask].copy()

    df_unach = df_unach.sort_values(
        by=["impact", "difficulty"], ascending=[False, True]
    )
    df_ach = df_ach.sort_values(by=["impact", "difficulty"], ascending=[False, True])

    tabs = st.tabs(["すべて表示", "未達成のみ", "達成済みのみ"])
    with tabs[0]:
        for _, row in df_unach.iterrows():
            quest_card(row, achieved=False, key_prefix="list_all_unach")
        if not df_ach.empty:
            st.markdown("## 達成済み（一覧の末尾）")
            for _, row in df_ach.iterrows():
                quest_card(row, achieved=True, key_prefix="list_all_ach")

    with tabs[1]:
        if df_unach.empty:
            st.success("未達成クエストはありません！")
        else:
            for _, row in df_unach.iterrows():
                quest_card(row, achieved=False, key_prefix="list_unach")

    with tabs[2]:
        if df_ach.empty:
            st.info("達成済みクエストはまだありません。")
        else:
            for _, row in df_ach.iterrows():
                quest_card(row, achieved=True, key_prefix="list_ach")


if __name__ == "__main__":
    main()
