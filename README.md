# GX with TQ レコメンド機能（Streamlit）

`gx_quests.csv` に入っている「クエスト」データをもとに、ユーザーの達成履歴から次におすすめのクエストを **5件** 提案する Streamlit アプリです。  
おすすめは「似ているものを優先しつつ、同じような候補ばかりにならないように」 **TF-IDF + MMR + 探索枠（exploration）** を組み合わせています。

---

## 使い方（アプリの操作）

### 1. おすすめクエストを見る
トップの「おすすめクエスト」に、未達成のおすすめが最大5件表示されます。

- **exploit 4件**：これまで達成した傾向に合うものを中心におすすめ
- **explore 1件**：あえて少し違う方向のクエストを混ぜる（マンネリ防止）

### 2. 達成したら「達成 ✅」を押す
各カード右上の **「達成 ✅」** を押すと達成済みとして記録され、画面が再計算されます。

### 3. クエスト一覧を見る
下の「クエスト一覧」はタブで切り替えできます。

- すべて表示 / 未達成のみ / 達成済みのみ

一覧は **未達成→達成済み** の順で、各グループ内は **impact（高い順）→ difficulty（低い順）** で並びます。

### 4.（任意）達成状況をリセット / 履歴CSVをダウンロード
サイドバーに以下があります。

- **達成状況をリセット**
- **達成履歴CSVをダウンロード**（`quest_id, achieved, timestamp`）

---

## レコメンドのロジック

### 概要

このアプリは、次の考え方でおすすめを作っています。

1. **まず候補を絞る**  
   - 繰り返しできるクエスト（`repeatable="はい"`）だけ  
   - すでに達成したクエストは外す

2. **「あなたに合いそう度」を点数化する**  
   点数は主に3つで決まります。
   - **似ている度**：過去に達成したクエストのタイトルと似ているか  
   - **達成できそう度**：あなたの達成率（全体・カテゴリ別）から見て成功しやすいか  
   - **インパクト**：インパクトが大きいクエストか

3. **同じようなおすすめばかりにならないように並び替える**  
   似た候補が続かないように「多様性」を考慮して上位4件を選びます（MMR）。

4. **最後に“探索枠”を1件混ぜる**  
   いつもと違うカテゴリ・ほどよい難易度のものを1件入れて、視野を広げます。

結果として、  
- 「いつもの自分に合うおすすめ」4件  
- 「新しい発見」1件  
の合計5件が出ます。

---

### 詳細

`gx_quest_recommender.py` の `recommend_quests()` に対応した説明です。

#### 0) 設定（デフォルト）
`RecommendConfig`（アプリ側では以下を使用）：

- `k_total=5`, `k_exploit=4`, `k_explore=1`
- `mmr_lambda=0.70`
- 重み：`w_sim=0.60`, `w_succ=0.20`, `w_imp=0.20`

#### 1) 候補集合の作成（Filtering）

候補 `m` は `quests` から以下で絞り込みます。

- `allow_repeatable_only=True` のため  
  `m = m[m["repeatable"] == "はい"]`
- 達成済み除外（`exclude_completed=True`）  
  `achieved_ids = {quest_id | user_history.achieved == True}`  
  `m = m[~m["quest_id"].isin(achieved_ids)]`

※ `context_location` は実装上対応していますが、アプリでは **`None` 固定**のためロケーションフィルタは使いません。

#### 2) 特徴量の計算

##### 2-1) テキスト類似度 `sim`
- 達成済みクエストの `title` を `achieved_titles` とする
- `title` を TF-IDF でベクトル化（文字 n-gram 2〜4, `char_wb`）
- 達成済みベクトル平均と候補ベクトルの cosine similarity を計算

\[
\mathrm{sim}_i = \cos\left(\frac{1}{|A|}\sum_{t \in A} \mathbf{v}(t),\ \mathbf{v}(title_i)\right)
\]

冷スタート（達成履歴がない）では `sim_i = 0`。

##### 2-2) 成功確率 `succ`
`compute_success_prob()` で、クエストごとに成功確率を推定します。

- 全体達成率  
\[
overall = mean(\mathrm{achieved})
\]
- カテゴリ別達成率  
\[
cat\_rate(c) = mean(\mathrm{achieved} \mid category=c)
\]
- 各クエスト \(i\) のカテゴリを \(c_i\) とすると  
\[
succ_i = 0.5 \cdot overall + 0.5 \cdot cat\_rate(c_i)
\]
履歴がない場合は default（0.5）。

##### 2-3) インパクト `imp`
`impact`（想定 1〜5）を利用。

#### 3) 0〜1正規化

実装の `_normalize_0_1()` により min-max 正規化します（全て同値なら 0）。

\[
x^{(n)} = \frac{x - \min(x)}{\max(x)-\min(x)}
\]

- `sim_n`, `succ_n`, `imp_n` を作成

#### 4) ベーススコア `base`

\[
base_i = w_{sim} \cdot sim^{(n)}_i + w_{succ} \cdot succ^{(n)}_i + w_{imp} \cdot imp^{(n)}_i
\]

デフォルト重み：
\[
base_i = 0.60\cdot sim^{(n)}_i + 0.20\cdot succ^{(n)}_i + 0.20\cdot imp^{(n)}_i
\]

#### 5) 多様性リランキング（MMR）で exploit を4件選ぶ

候補同士の類似度行列 `pairwise` を、候補タイトルだけで TF-IDF → cosine similarity で作ります。

MMR の選択値（実装の形）：

\[
val(i) = \lambda \cdot sim^{(n)}_i + (1-\lambda)\cdot base_i - (1-\lambda)\cdot \max_{j \in S} pairwise(i,j)
\]

- \(S\)：既に選んだ集合
- \(\lambda = 0.70\)
- 最初の1件は  
\[
\arg\max_i \left(\lambda \cdot sim^{(n)}_i + (1-\lambda)\cdot base_i\right)
\]
- 以降は上式の \(val(i)\) 最大を順に選び、`k_exploit=4` 件確定

選ばれた4件には `reason="exploit"` を付与し、出力列として
`similarity, success_prob, score(=base)` を持たせます。

#### 6) 探索枠 explore を1件選ぶ

exploit で選んだ4件を除いた `remaining` から、探索用スコア `explore_score` を作ります。

- 新規性：低類似度を優先  
  \((1 - rem\_sim^{(n)}_i)\)
- 未露出カテゴリを優先：履歴でのカテゴリ出現回数 `exposure` を min-max 正規化  
  \((1 - exposure^{(n)}_i)\)
- 難易度は3に近いほど良い（距離を負にして正規化）  
\[
diff\_pref_i = -|difficulty_i - 3|
\]
- impact も加点

最終探索スコア（実装通り）：
\[
explore\_score_i =
0.45(1-rem\_sim^{(n)}_i)+
0.25(1-exposure^{(n)}_i)+
0.15\cdot diff^{(n)}_i+
0.15\cdot imp^{(n)}_i
\]

その中で最大の1件を採用（`argmax`）し、`reason="explore"` を付与します。  
（現在は weighted random ではなく deterministic です）

#### 7) 出力
最終的に

- exploit 4件（MMRで選抜）
- explore 1件（探索枠）

を結合して返します。表示列は存在するものから優先順で整形されます。
