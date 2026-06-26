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

        start = time.time()

        response = requests.get(

            f"{API_URL}/search/{endpoint}",

            params={

                "query": query

            }

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

            st.markdown("---")

            for rank, doc in enumerate(
                results,
                start=1
            ):

                with st.container():

                    st.subheader(

                        f"{rank}. {doc['title']}"

                    )

                    col1, col2 = st.columns(
                        2
                    )

                    with col1:

                        st.write(
                            f"**Stance:** {doc['stance']}"
                        )

                    with col2:

                        st.write(
                            f"**Score:** {doc['score']:.4f}"
                        )

                    st.write(
                        doc["text"]
                    )

                    st.caption(
                        doc["doc_id"]
                    )

                    st.divider()