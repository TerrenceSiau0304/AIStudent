from langchain_tavily import TavilySearch


from core.config import get_settings

# Initialize the Tavily web search tool.
# It is used to search documents for web search generation
web_search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=get_settings().tavily_api_key,
)