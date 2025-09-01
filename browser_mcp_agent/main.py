import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from textwrap import dedent

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Page config
st.set_page_config(page_title="Pumpernickel", layout="wide")

# Custom CSS for grey and green theme
st.markdown(
    """
    <style>
    body, .main, .block-container {
        background-color: #f5f5f5 !important;
        color: #2c2c2c !important;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c2c2c;
        background: linear-gradient(90deg, #e8f5e8 0%, #d4edda 100%);
        padding: 24px 0 12px 0;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.5em;
        border: 2px solid #28a745;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #2c2c2c !important;
        border: 2px solid #28a745 !important;
        border-radius: 8px;
    }
    .stButton button {
        background-color: #28a745 !important;
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #218838 !important;
        transform: translateY(-1px);
    }
    .stSpinner > div > div {
        color: #28a745 !important;
    }
    .stMarkdown, .stCaption, .stText {
        color: #2c2c2c !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa !important;
        color: #ffffff !important;
        border-right: 3px solid #28a745;
    }
    .sidebar h3 {
        color: #28a745 !important;
        border-bottom: 2px solid #28a745;
        padding-bottom: 8px;
    }
    .sidebar .stMarkdown {
        color: #ffffff !important;
    }
    .sidebar .stCaption {
        color: #ffffff !important;
    }
    .help-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
    }
    .help-section h4 {
        color: #28a745 !important;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<div class='main-header'>Pumpernickel</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Hovershelf Product - Intelligent Web Browsing Agent</div>", unsafe_allow_html=True)

# Setup sidebar with example commands
with st.sidebar:
    st.markdown("<h3 style='color:#28a745;'>Example Commands</h3>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:#ffffff; font-weight:600;'>Navigation</p>", unsafe_allow_html=True)
    st.markdown("<span style='color:#ffffff;'>- Go to wikipedia.org/wiki/computer_vision</span>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:#ffffff; font-weight:600;'>Interactions</p>", unsafe_allow_html=True)
    st.markdown("<span style='color:#ffffff;'>- Click on the link to object detection and take a screenshot</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#ffffff;'>- Scroll down to view more content</span>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:#ffffff; font-weight:600;'>Multi-step Tasks</p>", unsafe_allow_html=True)
    st.markdown("<span style='color:#ffffff;'>- Navigate to wikipedia.org/wiki/computer_vision, scroll down, and report details</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#ffffff;'>- Scroll down and summarize the wikipedia page</span>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color:#28a745;'>", unsafe_allow_html=True)
    st.caption("Note: The agent uses Puppeteer to control a real browser.")

# Query input
query = st.text_area("Your Command", 
                   placeholder="Ask the agent to navigate to websites and interact with them")

# Initialize app and agent
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mcp_app = MCPApp(name="streamlit_mcp_agent")
    st.session_state.mcp_context = None
    st.session_state.mcp_agent_app = None
    st.session_state.browser_agent = None
    st.session_state.llm = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

# Setup function that runs only once
async def setup_agent():
    if not st.session_state.initialized:
        try:
            # Create context manager and store it in session state
            st.session_state.mcp_context = st.session_state.mcp_app.run()
            st.session_state.mcp_agent_app = await st.session_state.mcp_context.__aenter__()
            
            # Create and initialize agent
            st.session_state.browser_agent = Agent(
                name="browser",
                instruction="""You are a helpful web browsing assistant that can interact with websites using puppeteer.
                    - Navigate to websites and perform browser actions (click, scroll, type)
                    - Extract information from web pages 
                    - Take screenshots of page elements when useful
                    - Provide concise summaries of web content using markdown
                    - Follow multi-step browsing sequences to complete tasks
                    
                    When navigating, start with "www.lastmileai.dev" unless instructed otherwise.""",
                server_names=["puppeteer"],
            )
            
            # Initialize agent and attach LLM
            await st.session_state.browser_agent.initialize()
            st.session_state.llm = await st.session_state.browser_agent.attach_llm(OpenAIAugmentedLLM)
            
            # List tools once
            logger = st.session_state.mcp_agent_app.logger
            tools = await st.session_state.browser_agent.list_tools()
            logger.info("Tools available:", data=tools)
            
            # Mark as initialized
            st.session_state.initialized = True
        except Exception as e:
            return f"Error during initialization: {str(e)}"
    return None

# Main function to run agent
async def run_mcp_agent(message):
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: OpenAI API key not provided"
    
    try:
        # Make sure agent is initialized
        error = await setup_agent()
        if error:
            return error
        
        # Generate response without recreating agents
        # Switch use_history to False to reduce the passed context
        result = await st.session_state.llm.generate_str(
            message=message, 
            request_params=RequestParams(use_history=True)
            )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Run button
if st.button("Run Command", type="primary", use_container_width=True):
    with st.spinner("Processing your request..."):
        result = st.session_state.loop.run_until_complete(run_mcp_agent(query))
    
    # Display results
    st.markdown("<h4 style='color:#28a745;'>Response</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#ffffff;padding:16px;border-radius:8px;color:#2c2c2c;border:1px solid #28a745;'>{result}</div>", unsafe_allow_html=True)

# Display help text for first-time users
if 'result' not in locals():
    st.markdown(
        """
        <div class='help-section'>
        <h4 style='color:#28a745;'>How to use Pumpernickel:</h4>
        <ol>
            <li>Enter your OpenAI API key in your <code>.env</code> file</li>
            <li>Type a command for the agent to navigate and interact with websites</li>
            <li>Click 'Run Command' to see results</li>
        </ol>
        <p><strong style='color:#28a745;'>Capabilities:</strong></p>
        <ul>
            <li>Navigate to websites using Puppeteer</li>
            <li>Click on elements, scroll, and type text</li>
            <li>Take screenshots of specific elements</li>
            <li>Extract information from web pages</li>
            <li>Perform multi-step browsing tasks</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("<hr style='border-color:#28a745;'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#6c757d;'>Built with Streamlit, Puppeteer, and MCP-Agent Framework</div>", unsafe_allow_html=True)
