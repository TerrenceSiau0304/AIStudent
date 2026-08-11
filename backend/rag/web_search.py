from langchain_tavily import TavilySearch


from core.config import get_settings

web_search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=get_settings().tavily_api_key,
)