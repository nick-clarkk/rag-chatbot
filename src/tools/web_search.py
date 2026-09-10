from langchain_tavily import TavilySearch


web_search = TavilySearch(
    max_results=5, # Small enough to contend with search limits on Free Tier
    search_depth="basic", # Free tier search constraint (1 credit vs 2)
    include_answer=False, # Terra is responsible for the answer
    include_raw_content=False, # We don't need an entire webpage
)