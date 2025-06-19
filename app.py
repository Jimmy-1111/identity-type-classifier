import streamlit as st
import pandas as pd
import os
import tempfile
from sentence_transformers import SentenceTransformer, util

# === 使用日文語意模型 ===
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

# === 日文定義語句 ===
default_definitions = {
    "アイデンティティ挑戦型イノベーション": "企業が「○○に転換する」と明確に宣言し、「私たちは何者か」を再定義する。",
    "アイデンティティ拡張型イノベーション": "企業が既存の価値や理念を新しい領域に応用し、連続性と融合性を強調する。",
    "アイデンティティ強化型イノベーション": "既存のアイデンティティと一致したイノベーションを推進するが、アイデンティティの言語や定位の調整は伴わない。",
    "伝統的／中立的言語": "品質・コスト・効率など運営に関する内容のみを記述し、アイデンティティや価値に関する意味が全く含まれていない。",
    "その他（Other）": "アイデンティティ、イノベーション、価値、使命に関わらない文で、上記4タイプとの意味的な類似性が明らかに低い。"
}

# === 頁面設定 ===
st.set_page_config(page_title="日本語・年報文分類", layout="centered")
st.title("📊 日本語：企業年報文のアイデンティティ分類")

# === 類型定義調整 ===
st.header("📝 分類基準の定義文を確認・変更できます")
category_inputs = {}
for cat, default in default_definitions.items():
    category_inputs[cat] = st.text_area(f"{cat}", value=default, height=70)

# === 檔案上傳 ===
st.header("📂 年報語句を含む Excel ファイルをアップロード")
uploaded_files = st.file_uploader("複数のファイルを選択できます", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_excel(uploaded_file, sheet_name=0)

        if "語句内容" not in df.columns:
            st.warning(f"❗ ファイル {uploaded_file.name} に '語句内容' 列が見つかりません。")
            continue

        sentences = df["語句内容"].astype(str).tolist()

        # === 句子向量化 ===
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        definition_embeddings = {
            label: model.encode(text, convert_to_tensor=True)
            for label, text in category_inputs.items()
        }

        # === 分類判定 ===
        predicted_labels = []
        similarity_scores = []
        for sent_emb in sentence_embeddings:
            scores = {
                label: float(util.cos_sim(sent_emb, def_emb))
                for label, def_emb in definition_embeddings.items()
            }
            best_label = max(scores, key=scores.get)
            predicted_labels.append(best_label)
            similarity_scores.append(scores[best_label])

        df["分類ラベル"] = predicted_labels
        df["類似度スコア"] = similarity_scores

        # === 結果輸出 ===
        filename = uploaded_file.name.replace(".xlsx", "_分類結果.xlsx")
        output_path = os.path.join(tempfile.gettempdir(), filename)
        df.to_excel(output_path, index=False)

        with open(output_path, "rb") as f:
            st.download_button(
                label=f"📥 結果をダウンロード：{filename}",
                data=f,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
