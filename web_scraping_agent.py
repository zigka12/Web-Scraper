import os
from openai import AzureOpenAI
from bs4 import BeautifulSoup as Soup
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
import io
import base64
from PIL import Image
import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import base64
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from tavily import TavilyClient
from langchain.chains import ConversationalRetrievalChain
# from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts.chat import ChatPromptTemplate
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from collections import defaultdict
from langchain_community.retrievers import BM25Retriever
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from PIL import Image
import io
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from urllib.parse import urljoin
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import undetected_chromedriver as uc
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from fastapi import FastAPI, UploadFile, File, Form, Response, Header
from typing import Union, List, Annotated
from collections import deque
import json
import networkx as nx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st


# Priority: st.secrets (on Streamlit Cloud), fallback to local .env

api_key = st.secrets["OPENAI_API_KEY"]
tavily_key = st.secrets["TAV_API_KEY"]
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4o",
    temperature=0.7
)

embeddings = OpenAIEmbeddings(
    openai_api_key=api_key,
    model="text-embedding-ada-002"
)

client = openai.OpenAI(api_key=api_key)
tavily_client = TavilyClient(api_key=tavily_key)
os.environ["USER_AGENT"] = "MyWebAgent/1.0"



def search_results(query):

    response = tavily_client.search(query,max_results=5)
    return response


def transform_data_list(data_list):
    return [
        {
            'text': f"{data.get('title', '')} - {data.get('content', '')}",
            'url': data.get('url', ''),
            'link_relevancy_score': data.get('score', 0)
        }
        for data in data_list
    ]


def transform_data(data_list):
    return [ f"{data.get('title', '')} - {data.get('content', '')}"
          
        for data in data_list
    ]


def get_base_url(url):
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    return base_url


def extract_links_from_html(html, url):
    base_url = get_base_url(url)
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        text = a.get_text(strip=True) or ""
        full_url = urljoin(base_url, a['href'])
        links.append({'text': text, 'link': full_url})
    
    page_text = soup.get_text(separator="\n", strip=True)
    if page_text is None :
        page_text=""

    return links, page_text



def scrape_full_page_screenshots(url, scroll_pause=2):
    driver = uc.Chrome()
    driver.get(url)

    # Let initial content load
    time.sleep(scroll_pause)

    # Try hiding common popups (sign-in modals, overlays, etc.)
    driver.execute_script("""
        const selectors = [
            '.popup', '.modal', '.overlay', '#login-popup', '#sign-in', '.signin', '.backdrop'
        ];
        selectors.forEach(selector => {
            const el = document.querySelector(selector);
            if (el) {
                el.style.display = 'none';
                el.remove(); // optional: fully remove it from DOM
            }
        });
    """)
    time.sleep(1)  # wait for DOM changes to take effect

    html = driver.page_source
    screenshot_images = []

    # Get scroll and viewport height
    total_height = driver.execute_script("return document.body.scrollHeight")
    viewport_height = driver.execute_script("return window.innerHeight")

    scroll_position = 0
    captured = set()

    while scroll_position < total_height:
        if scroll_position in captured:
            break
        captured.add(scroll_position)

        # Screenshot current viewport
        png_data = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png_data))
        screenshot_images.append(img.copy())

        # Scroll down
        scroll_position += viewport_height
        driver.execute_script(f"window.scrollTo(0, {scroll_position})")
        time.sleep(scroll_pause)

        # Recheck height in case it's infinite-scroll
        total_height = driver.execute_script("return document.body.scrollHeight")

    driver.quit()
    return html, screenshot_images

def pil_images_to_base64(image_list):
    base64_list = []
    for img in image_list:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        base64_list.append(b64_str)
    return base64_list

def evaluate_page_with_llm(llm, query, base64_screenshots, page_text, url):
    prompt = f"""
        You are a web browsing assistant helping to explore pages relevant to a user query.

        User Query:
        {query}

        Page URL:
        {url}

        Page Text:
        {page_text[:2500]}

        Based on the page content, determine:
        1. Summarize the content of the page.
        2. How relevant is this page to the user query? Give a score from 0 to 10.
        3. Decide whether to explore this page's sublinks in more depth (DFS) or move on to other pages (BFS). 
            - Choose **DFS** if the user's query is only partially answered and there is a reasonable chance the sublinks on the page may contain relevant information.
            - Choose **BFS** if the page already answers the query well or it's unlikely that sublinks will add value.
        Respond strictly in JSON format like:
        {{
        "Page Summary": "<summary>",
        "relevancy_score": <Number between 0 and 10>,
        "strategy": "<DFS or BFS>"
        }}
    """
    content = [{"type": "text", "text": prompt}]
    for img_str in base64_screenshots[:3]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_str}",
                "detail": "high"
            }
        })
    messages = [
        {"role": "system", "content": "You are a helpful web browsing assistant. Respond in plain JSON without markdown formatting or code blocks."},
        {"role": "user", "content": content},
    ]

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        temperature=0
    )

    assistant_message = response.choices[0].message.content.strip()

    # Remove code block markers if present
    if assistant_message.startswith("```json"):
        assistant_message = assistant_message[len("```json"):].strip()
    elif assistant_message.startswith("```"):
        assistant_message = assistant_message[len("```"):].strip()
    if assistant_message.endswith("```"):
        assistant_message = assistant_message[:-3].strip()

    try:
        result = json.loads(assistant_message)
        return result.get("relevancy_score", 0), result.get("strategy", "BFS"), result.get("Page Summary")
    except Exception as e:
        print("LLM JSON parse error:", e)
        print("Raw assistant message:", assistant_message)
        return 0.0, "BFS", ""




def traverse_web(query, max_depth=2):
    results = search_results(query)
    pages = transform_data_list(results['results'])
    queue = deque()
    visited = set()
    page_info = {}

    # Enqueue initial results
    for page in pages:
        queue.append((page['url'], 0, None))  # (url, depth, parent)
 
    while queue:
        url, depth, parent = queue.popleft()
        if url in visited or depth > max_depth:
            continue

        print(f"Visiting: {url} | Depth: {depth}")
        visited.add(url)

        try:
            html, screenshots = scrape_full_page_screenshots(url)
            base64_imgs = pil_images_to_base64(screenshots)
            sublinks, page_text = extract_links_from_html(html, url)
            relevancy_score, strategy, page_summary = evaluate_page_with_llm(client, query, base64_imgs, page_text, url)
        except Exception as e:
            print(f"Failed processing {url}: {e}")
            continue

        # Save metadata
        page_info[url] = {
            "url": url,
            "parent": parent,
            "relevancy_score": relevancy_score,
            "strategy": strategy,
            "depth": depth,
            "Page_Summary" : page_summary
        }

        # Handle DFS
        if strategy == "DFS":
            top_links = [link for link in sublinks if link["link"] not in visited][:2]
            for link in top_links:
                link_url = link["link"]
                if link_url not in visited:
                    queue.appendleft((link_url, depth + 1, url))  # DFS: insert at front
        # else:
        #     for link in sublinks:
        #         link_url = link["link"]
        #         if link_url not in visited:
        #             queue.append((link_url, depth + 1, url))  # BFS: insert at end

    return page_info

def build_graph(page_info):
    G = nx.DiGraph()
    
    for url, info in page_info.items():
        G.add_node(url)  # Add the URL as a node
        parent = info.get('parent')
        if parent is not None and parent in page_info:
            G.add_edge(parent, url)  # Create an edge from parent to current page
    
    return G

def compute_pagerank(graph):
    return nx.pagerank(graph)

def update_page_info_with_pagerank(page_info, pagerank_scores):
    for url in page_info:
        page_info[url]['pagerank'] = pagerank_scores.get(url, 0.0)
    return page_info

def chunk_page_info(page_info, chunk_size=5):
    items = list(page_info.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i + chunk_size])

def summarize_page_info_with_llm(page_info, query, client, model='gpt-4o', max_text_length=4000):

    # Convert page_info to JSON string with indentation for readability
    page_info_str = json.dumps(page_info, indent=2)

    # Truncate if too long to keep prompt manageable (avoid token limits)
    if len(page_info_str) > max_text_length:
        page_info_str = page_info_str[:max_text_length] + "\n...TRUNCATED..."

    prompt = f"""
You are an expert web research assistant. Given the user query and detailed info about many web pages including their relevance scores(assigned by an LLM), page summaries, and PageRank scores(from PageRank algorithm), provide:

1. A detailed and comprehensive overall summary of what you found.
2. Recommendations on which pages or subtopics to explore further.
3. Any other relevant insights or suggestions.

User Query:
{query}

Pages Info:
{page_info_str}

Respond clearly and concisely. Do not mention any of the scores that you are given
"""

    messages = [
        {"role": "system", "content": "You are a helpful web browsing assistant."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content

def workflow(query):
    page_info = traverse_web(query)
    url_graph = build_graph(page_info)
    pagerank_scores = compute_pagerank(url_graph)
    page_info_updated = update_page_info_with_pagerank(page_info, pagerank_scores)
    final_response = summarize_page_info_with_llm(page_info_updated, query, client)
    return final_response

