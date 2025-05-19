import streamlit as st
from web_scraping_agent import workflow, search_results

st.set_page_config(page_title="Web Scraping AI Agent", layout="wide")

st.title("🔍 Web Scraping AI agent")

query = st.text_input("💡 Enter your query here:")

if st.button("🚀 Run Workflow"):
    if query:
        with st.spinner("🔎 Searching the web for relevant URLs..."):
            links = search_results(query)
            urls = [result.get('url', '') for result in links['results']]
            # urls = search_web_for_query(query)

        if urls:
            st.success(f"🌐 Found {len(urls)} URLs to explore!")
            with st.expander("📜 See URLs we're exploring"):
                for i, url in enumerate(urls, start=1):
                    st.markdown(f"{i}. [{url}]({url})")

            with st.spinner("🧠 Scraping the web and running LLM analysis..."):
                result = workflow(query)

            if result:
                st.success("✅ Answer generated!")
                st.markdown(f"### 📖 Answer\n{result}")
            else:
                st.error("❌ No relevant information found.")

        else:
            st.error("❌ No URLs found for the query.")
    else:
        st.warning("⚠️ Please enter a query to proceed.")
