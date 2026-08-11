from langchain_tavily import TavilySearch


from core.config import get_settings

def web_search_tool():
    return TavilySearch(tavily_api_key = get_settings().tavily_api_key)
