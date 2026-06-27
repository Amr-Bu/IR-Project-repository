import time
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Information Retrieval System",
    page_icon="🔎",
    layout="wide"
)


st.title("🔎 Information Retrieval System")

st.markdown(
    "Search documents using TF-IDF, BM25, Semantic Search or Hybrid Search."
)

st.sidebar.header("Search Settings")

method = st.sidebar.selectbox(
    "Search Method",
    [
        "TF-IDF",
        "BM25",
        "Semantic",
        "Hybrid"
    ]
)

top_k = st.sidebar.selectbox(
    "Top K",
    [
        5,
        10,
        20
    ],
    index=1
)

alpha = 0.7

if method == "Hybrid":

    alpha = st.sidebar.slider(
        "Hybrid Alpha (BM25 weight)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    st.sidebar.caption(
        f"BM25 weight: {alpha:.1f} | Semantic weight: {1 - alpha:.1f}"
    )

refine = st.sidebar.checkbox(
    "Enable Query Refinement (Synonym Expansion)",
    value=False
)

if method == "Semantic":

    st.sidebar.caption(
        "ℹ️ Refinement doesn't apply to Semantic search "
        "(embeddings already capture synonyms)."
    )

query = st.text_input(
    "Search Query"
)

button = st.button(
    "🔍 Search"
)

if button:

    if query.strip() == "":

        st.warning(
            "Please enter a search query."
        )

    else:

        endpoint = {

            "TF-IDF": "tfidf",

            "BM25": "bm25",

            "Semantic": "semantic",

            "Hybrid": "hybrid"

        }[method]

        params = {
            "query": query
        }

        if method == "Hybrid":

            params["alpha"] = alpha

        if method in ("TF-IDF", "BM25", "Hybrid"):

            params["refine"] = refine

        start = time.time()

        response = requests.get(

            f"{API_URL}/search/{endpoint}",

            params=params

        )

        elapsed = time.time() - start

        if response.status_code != 200:

            st.error(
                "API Error."
            )

        else:

            results = response.json()

            st.success(
                f"Found {len(results)} Results"
            )

            st.info(
                f"Search Time : {elapsed:.3f} sec"
            )

            topic_response = requests.get(

                f"{API_URL}/topic/detect",

                params={
                    "query": query
                }

            ).json()

            st.info(
                f"🏷️ Detected Query Topic: **{topic_response['topic']}**"
            )

            st.markdown("---")

            for rank, doc in enumerate(
                results,
                start=1
            ):

                with st.container():

                    st.subheader(

                        f"{rank}. {doc['title']}"

                    )

                    col1, col2, col3 = st.columns(
                        3
                    )

                    with col1:

                        st.write(
                            f"**Stance:** {doc['stance']}"
                        )

                    with col2:

                        st.write(
                            f"**Score:** {doc['score']:.4f}"
                        )

                    with col3:

                        st.write(
                            f"**Topic:** {doc['topic']}"
                        )

                    st.write(
                        doc["text"]
                    )

                    st.caption(
                        doc["doc_id"]
                    )

                    st.divider()