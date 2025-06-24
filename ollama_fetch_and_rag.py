import streamlit as st

# Create tabs
tab1, tab2 = st.tabs(["📥 Extract Reddit Links", "🤖 RAG on Reddit Posts"])


with tab1:
    import praw
    import os

    st.subheader("🔗 Reddit Post Link Extractor")
    st.write("Enter your Reddit API credentials to extract links from a subreddit.")

    # Ask what the user is fetching (e.g., 'nas_reviews', 'qnap_issues')
    custom_folder_name = st.text_input("📁 What are you going to fetch? (Folder name)", value="qnap_posts")

    client_id_1 = st.text_input("Client ID", key="client_id_1")
    client_secret_1 = st.text_input("Client Secret", type="password", key="client_secret_1")
    user_agent_1 = st.text_input("User Agent", key="user_agent_1")
    subreddit_name = st.text_input("Subreddit Name", value="qnap", key="subreddit_name")
    limit = st.number_input("Number of Posts to Fetch", min_value=1, max_value=1000, value=100)

    if st.button("Fetch Links"):
        if not all([client_id_1, client_secret_1, user_agent_1, subreddit_name, custom_folder_name]):
            st.error("Please fill in all required fields including folder name.")
        else:
            try:
                folder_path = os.path.join("posts", custom_folder_name)
                os.makedirs(folder_path, exist_ok=True)

                reddit = praw.Reddit(
                    client_id=client_id_1,
                    client_secret=client_secret_1,
                    user_agent=user_agent_1
                )
                subreddit = reddit.subreddit(subreddit_name)
                post_links = [
                    f"https://www.reddit.com{post.permalink}"
                    for post in subreddit.hot(limit=limit)
                ]

                filename = os.path.join(folder_path, f"{subreddit_name}_posts.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("\n".join(post_links))

                st.success(f"✅ {len(post_links)} links saved to `{filename}`")

                with open(filename, "rb") as f:
                    st.download_button(
                        label="Download Link File",
                        data=f,
                        file_name=f"{subreddit_name}_posts.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"❌ Error: {e}")


with tab2:
    import asyncio
    import sys
    import nest_asyncio

    if sys.platform == "win32":
        if sys.version_info >= (3, 8) and sys.version_info < (3, 9):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        nest_asyncio.apply()

    import warnings
    warnings.filterwarnings("ignore", message="missing ScriptRunContext")

    import os
    os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

    import streamlit as st
    from langchain_community.document_loaders import UnstructuredURLLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document
    import praw
    import time
    import torch
    import chromadb
    from ollama import chat
    from typing import List

    chromadb.api.client.SharedSystemClient.clear_system_cache()
    torch.classes.__path__ = []

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'reddit_contents' not in st.session_state:
        st.session_state.reddit_contents = []

    def process_text_files(folder_path: str) -> List[str]:
        urls = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    urls.extend([line.strip() for line in f if line.strip()])
        return urls

    def fetch_reddit_posts(reddit_client, url: str):
        try:
            submission = reddit_client.submission(url=url)
            return f"Title: {submission.title}\nContent: {submission.selftext}"
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def create_vector_database(reddit, folder_path):
        reddit_urls = process_text_files(folder_path)
        if not reddit_urls:
            st.error("No Reddit URLs found.")
            return None, None

        st.write(f"Found {len(reddit_urls)} URLs.")
        contents = []

        for url in reddit_urls[:50]:
            content = fetch_reddit_posts(reddit, url)
            if content:
                contents.append(content)
            time.sleep(1)

        if not contents:
            st.error("Failed to fetch Reddit posts.")
            return None, None

        docs = [Document(page_content=c) for c in contents]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )

        vectorstore = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        #vectorstore.persist()

        return vectorstore, contents

    st.title("🔎 Reddit RAG with Ollama (Gemma3)")

    st.subheader("Reddit API Configuration")
    client_id = st.text_input("Reddit Client ID")
    client_secret = st.text_input("Reddit Client Secret", type="password")
    user_agent = st.text_input("User Agent", value="RedditScraper/1.0")
    folder_path = st.text_input("Folder with .txt Reddit URLs", "posts")

    if st.button("Create Vector Database"):
        if not all([client_id, client_secret, user_agent]):
            st.warning("Enter all API credentials.")
            st.stop()

        if not os.path.exists(folder_path):
            st.error("Folder path doesn't exist.")
            st.stop()

        with st.spinner("Processing..."):
            try:
                reddit = praw.Reddit(client_id=client_id,
                                     client_secret=client_secret,
                                     user_agent=user_agent)
                vectorstore, contents = create_vector_database(reddit, folder_path)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.reddit_contents = contents
                    st.success("Vector database created successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    # Try to reload vectorstore from disk if it's missing from session
    if st.session_state.get("vectorstore") is None and os.path.exists("chroma_db"):
        st.session_state.vectorstore = Chroma(
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            ),
            persist_directory="chroma_db"
        )

    if st.session_state.vectorstore:
        st.subheader("Ask Questions")
        user_query = st.text_input("Your question about the Reddit posts")

        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    retrieved_docs = retriever.invoke(user_query)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                    prompt = f"""
You are a helpful assistant analyzing Reddit posts.

Answer the following question **only** based on the given Reddit context.

Respond in **detailed bullet points** covering multiple perspectives, insights, or steps. Your answer should be **at least 250 words**. If the answer cannot be derived from the context, respond with: "The context does not contain enough information to answer this."

### Context:
{context}

### Question:
{user_query}

### Detailed Answer (in point form):
"""

                    response = chat(model='gemma3:4b', messages=[
                        {"role": "user", "content": prompt}
                    ])

                    st.subheader("Answer:")
                    st.write(response['message']['content'])

                    st.subheader("🔍 Relevant Reddit Posts Used:")
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Post {i+1}**")
                        st.write(doc.page_content[:500] + "...")
                        st.markdown("---")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Please create the vector database first.")
