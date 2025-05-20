# AI-Powered Web Browsing & Research Agent

A fully autonomous web-browsing assistant that performs intelligent query-driven web research. It fetches search results, scrapes full-page screenshots, extracts page content and links, evaluates relevance via GPT-4o, and traverses the web adaptively using DFS/BFS strategies. It builds a page network graph, computes PageRank, and finally delivers a comprehensive, human-readable summary of findings.

---

## Workflow description

- Automated Web Search — Queries Tavily’s web search API for relevant starting URLs.
- Full Page Screenshots — Uses Selenium + undetected-chromedriver to capture full-page screenshots as the page scrolls.
- Page Content Extraction — Extracts page text and all hyperlinks.
- LLM-Based Page Evaluation — GPT-4o evaluates each page for:
  - Summary of content  
  - Relevance score (0-10)  
  - Exploration strategy (DFS/BFS)
- Dynamic Web Traversal — Explores the web using an adaptive queue-based BFS/DFS approach.
- Graph Construction & PageRank — Constructs a network graph of visited pages and computes PageRank scores for influence ranking.
- Final Summarization — Uses GPT-4o to deliver a comprehensive, query-specific summary of all findings and recommendations.

---

## Tech Stack

- Python 3.10+
- LangChain
- OpenAI GPT-4o API
- Tavily Search API
- Selenium + undetected-chromedriver
- BeautifulSoup
- NetworkX (PageRank)
- Pillow (image handling)
- FastAPI (optional server-side API exposure)
- dotenv for environment management

---

## Architecture & Workflow

```mermaid
graph TD
A[User Query] --> B[Tavily Search API]
B --> C[Initial URLs]
C --> D[Scrape Full Page Screenshot]
D --> E[Extract Page Text & Links]
E --> F[LLM Evaluation: Summary + Relevance + Strategy]
F -->|DFS/BFS| C
F --> G[Build Network Graph]
G --> H[Compute PageRank]
H --> I[Summarize Findings with GPT-4o]
I --> J[Final Summary Report]
