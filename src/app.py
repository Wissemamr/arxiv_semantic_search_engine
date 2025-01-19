import streamlit as st
from search_engine import SemanticSearchEngine
from utils import clean_text


def main():
    st.title("Research Paper Semantic Search")
    # col 1 on the left is for the query adn col2 on the right is for the slide bar (top k)
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your search query:", "")
    with col2:
        top_k = st.slider("Results", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        query = clean_text(query)
        if query:
            try:
                search_engine = SemanticSearchEngine()
                results = search_engine.search(query, top_k=top_k)
                if results:
                    st.success(f"Found {len(results)} results")
                    for res in results:
                        title = res["payload"].get("title", "N/A")
                        abstract = res["payload"].get("abstract", "N/A")
                        score = f"{res['score']:.4f}"

                        st.markdown(
                            f"""
                            <div style="border:1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px;">
                                <h3 style="margin-bottom: 5px;">{title}</h3>
                                <p style="font-size: 14px; margin-bottom: 10px;">{abstract}</p>
                                <p style="color: #555;"><b>Score:</b> {score}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("No results found for the given query.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query to search.")


if __name__ == "__main__":
    main()
