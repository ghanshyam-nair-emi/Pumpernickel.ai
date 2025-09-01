import streamlit as st
import os
from mem0 import Memory
from multion.client import MultiOn
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="ARXIV Search", layout="wide")

# Custom CSS for grey and white theme
st.markdown(
    """
    <style>
    body, .main, .block-container {
        background-color: #f8f9fa !important;
        color: #2c2c2c !important;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c2c2c;
        background: linear-gradient(90deg, #e9ecef 0%, #dee2e6 100%);
        padding: 20px 0 16px 0;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #2c2c2c !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px;
    }
    .stButton button {
        background-color: #6c757d !important;
        color: #ffffff !important;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #5a6268 !important;
    }
    .stSpinner > div > div {
        color: #6c757d !important;
    }
    .stMarkdown, .stText {
        color: #2c2c2c !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa !important;
        color: #2c2c2c !important;
    }
    .api-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 1.5rem;
    }
    .search-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<div class='main-header'>ARXIV Search</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered research assistant with memory for academic paper discovery</div>", unsafe_allow_html=True)

# API Keys Section
st.markdown("<div class='api-section'>", unsafe_allow_html=True)
st.markdown("#### API Configuration")
st.markdown("Enter your API keys to get started:")

col1, col2 = st.columns(2)
with col1:
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
with col2:
    multion_key = st.text_input("MultiOn API Key", type="password", placeholder="Enter your MultiOn API key")

api_keys = {'openai': openai_key, 'multion': multion_key}
st.markdown("</div>", unsafe_allow_html=True)

if all(api_keys.values()):
    os.environ['OPENAI_API_KEY'] = api_keys['openai']
    # Initialize Mem0 with Qdrant
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "model": "gpt-4o-mini",
                "host": "localhost",
                "port": 6333,
            }
        },
    }
    memory, multion, openai_client = Memory.from_config(config), MultiOn(api_key=api_keys['multion']), OpenAI(api_key=api_keys['openai'])

    # User Configuration
    st.sidebar.markdown("### User Settings")
    user_id = st.sidebar.text_input("Username", placeholder="Enter your username")
    
    # Search Section
    st.markdown("<div class='search-section'>", unsafe_allow_html=True)
    st.markdown("#### Research Paper Search")
    search_query = st.text_input("Search Query", placeholder="Enter your research query (e.g., 'machine learning in healthcare')")
    st.markdown("</div>", unsafe_allow_html=True)

    def process_with_gpt4(result):
        """Processes an arXiv search result to produce a structured markdown output.

    This function takes a search result from arXiv and generates a markdown-formatted
    table containing details about each paper. The table includes columns for the 
    paper's title, authors, a brief abstract, and a link to the paper on arXiv. 

    Args:
        result (str): The raw search result from arXiv, typically in a text format.

    Returns:
        str: A markdown-formatted string containing a table with paper details."""
        prompt = f"""
        Based on the following arXiv search result, provide a proper structured output in markdown that is readable by the users. 
        Each paper should have a title, authors, abstract, and link.
        Search Result: {result}
        Output Format: Table with the following columns: [{{"title": "Paper Title", "authors": "Author Names", "abstract": "Brief abstract", "link": "arXiv link"}}, ...]
        """
        response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return response.choices[0].message.content

    # Search Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button('Search for Papers', type="primary", use_container_width=True)
    
    if search_button and search_query:
        with st.spinner('Searching and processing papers...'):
            relevant_memories = memory.search(search_query, user_id=user_id, limit=3)
            prompt = f"Search for arXiv papers: {search_query}\nUser background: {' '.join(mem['text'] for mem in relevant_memories)}"
            result = process_with_gpt4(multion.browse(cmd=prompt, url="https://arxiv.org/"))
            
            # Results Section
            st.markdown("---")
            st.markdown("### Search Results")
            st.markdown(result)
    elif search_button and not search_query:
        st.warning("Please enter a search query.")

    # Memory Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Memory Management")
    if st.sidebar.button("View Stored Memory"):
        memories = memory.get_all(user_id=user_id)
        if memories:
            st.sidebar.markdown("**Stored Information:**")
            for i, mem in enumerate(memories, 1):
                st.sidebar.markdown(f"{i}. {mem['text']}")
        else:
            st.sidebar.info("No memories stored yet.")

else:
    # Warning message
    st.markdown(
        """
        <div style='padding: 20px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; color: #856404;'>
        <h4 style='color: #856404; margin-bottom: 10px;'>Configuration Required</h4>
        <p>Please enter both OpenAI and MultiOn API keys to use ARXIV Search.</p>
        <ul>
            <li><strong>OpenAI API Key:</strong> Required for AI processing of search results</li>
            <li><strong>MultiOn API Key:</strong> Required for web browsing capabilities</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    Built with Streamlit, Mem0, MultiOn, and OpenAI
    </div>
    """,
    unsafe_allow_html=True
)