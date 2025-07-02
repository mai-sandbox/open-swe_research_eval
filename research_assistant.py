"""
Advanced Research Assistant using LangGraph with custom state, memory, and human-in-the-loop.
Contains deliberate bugs to test LangGraph API knowledge.
"""

import os
from typing import Annotated, TypedDict, List, Optional, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import NodeInterrupt

load_dotenv()

# Custom state with research tracking
class ResearchState(TypedDict):
    # BUG 1: Wrong annotation for messages - should be Annotated[list, add_messages]
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Custom state fields for research tracking
    research_query: str
    research_progress: List[str]
    sources_found: List[str]
    requires_approval: bool
    approved_by_human: bool
    
    # BUG 2: Missing operator for custom state fields
    # Should use Annotated with proper operators for list fields
    summary: str

# Research tools
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Mock search results
    search_results = {
        "climate change": "Recent studies show global temperatures rising 1.1°C since pre-industrial times...",
        "ai research": "Latest breakthroughs in transformer models and multimodal AI systems...", 
        "quantum computing": "IBM and Google achieve quantum supremacy with 1000+ qubit systems...",
        "space exploration": "NASA's Artemis program targets moon landing by 2026..."
    }
    
    for keyword in search_results:
        if keyword in query.lower():
            return f"Search results for '{query}':\n{search_results[keyword]}"
    
    return f"No specific results found for '{query}'. General information available."

@tool
def document_lookup(document_id: str) -> str:
    """Look up information from internal documents."""
    docs = {
        "DOC-001": "Internal research on renewable energy shows 40% efficiency gains...",
        "DOC-002": "Market analysis indicates strong growth in AI sector...",
        "DOC-003": "Technical specifications for quantum encryption protocols..."
    }
    return docs.get(document_id, f"Document {document_id} not found in database")

@tool
def calculate_stats(expression: str) -> str:
    """Perform calculations and statistical analysis."""
    try:
        # Simple calculator (in real implementation, would use proper math library)
        result = eval(expression)  # Note: eval is unsafe, just for demo
        return f"Calculation result: {expression} = {result}"
    except:
        return f"Cannot calculate: {expression}"

@tool  
def request_human_approval(reason: str) -> str:
    """Request human approval for sensitive research topics."""
    return f"Human approval requested: {reason}. Waiting for approval..."

# Initialize model
model = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.1
)

tools = [web_search, document_lookup, calculate_stats, request_human_approval]

# BUG 3: Wrong method name - should be bind_tools (not bind_tool)
model_with_tools = model.bind_tool(tools)

def research_agent(state: ResearchState):
    """Main research agent that processes queries and calls tools."""
    
    system_prompt = """You are an advanced research assistant. You can:
    
    1. Search the web for information
    2. Look up internal documents  
    3. Perform calculations
    4. Request human approval for sensitive topics
    
    For sensitive topics (politics, controversial science, personal data), 
    you MUST request human approval before proceeding.
    
    Track your research progress and provide comprehensive summaries."""
    
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages)
    
    # Update research progress
    new_progress = state["research_progress"] + [f"Agent responded: {response.content[:50]}..."]
    
    return {
        "messages": [response],
        "research_progress": new_progress
    }

def execute_tools(state: ResearchState):
    """Execute tools and update research state."""
    # BUG 4: Wrong ToolNode instantiation - should pass tools directly, not as tools_list
    tool_node = ToolNode(tools_list=tools)
    
    # Execute tools
    result = tool_node.invoke(state)
    
    # BUG 5: Incorrect state update - not handling tool results properly
    # Should extract sources and update state fields
    last_message = result["messages"][-1]
    
    return {
        "messages": result["messages"],
        "sources_found": state["sources_found"] + ["tool_result"],
        "research_progress": state["research_progress"] + ["Tools executed"]
    }

def check_approval_needed(state: ResearchState) -> bool:
    """Check if human approval is needed based on tool calls."""
    last_message = state["messages"][-1]
    
    # BUG 6: Wrong attribute name - should be tool_calls not tools_called
    if hasattr(last_message, 'tools_called'):
        for tool_call in last_message.tools_called:
            if tool_call.get("name") == "request_human_approval":
                return True
    return False

def request_approval(state: ResearchState):
    """Handle human approval workflow."""
    approval_message = AIMessage(
        content="I need human approval to proceed with this research topic. Please review and approve."
    )
    
    # BUG 7: Wrong way to interrupt - should use interrupt() function correctly
    NodeInterrupt("Human approval required")
    
    return {
        "messages": [approval_message],
        "requires_approval": True
    }

def route_after_agent(state: ResearchState) -> Literal["tools", "approval", "summarize", "end"]:
    """Route conversation based on agent's response."""
    last_message = state["messages"][-1]
    
    # Check for tool calls
    # BUG 8: Wrong tool_calls attribute name
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Check if approval is needed
        if check_approval_needed(state):
            return "approval"
        return "tools"
    
    # Check if we should summarize (after multiple research steps)
    if len(state["research_progress"]) > 5:
        return "summarize"
        
    return "end"

def summarize_research(state: ResearchState):
    """Summarize the research findings."""
    summary_prompt = f"""
    Summarize the research conducted:
    
    Query: {state['research_query']}
    Progress: {state['research_progress']}
    Sources: {state['sources_found']}
    
    Provide a comprehensive summary.
    """
    
    summary_response = model.invoke([HumanMessage(content=summary_prompt)])
    
    return {
        "messages": [summary_response],
        "summary": summary_response.content
    }

# BUG 9: Wrong graph class - should be StateGraph not WorkflowGraph
workflow = WorkflowGraph(ResearchState)

# Add nodes
workflow.add_node("agent", research_agent)
workflow.add_node("tools", execute_tools)
workflow.add_node("approval", request_approval)
workflow.add_node("summarize", summarize_research)

# Add edges
workflow.add_edge(START, "agent")

# BUG 10: Wrong conditional edge setup - missing proper routing
workflow.add_conditional_edge(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "approval": "approval", 
        "summarize": "summarize",
        "end": END
    }
)

# Add edges back to agent
workflow.add_edge("tools", "agent")
# BUG 11: Missing edge from approval back to agent
# Should have: workflow.add_edge("approval", "agent")

workflow.add_edge("summarize", END)

# Setup memory with SQLite
# BUG 12: Wrong parameter name - should be conn_string not db_path
checkpointer = SqliteSaver.from_conn_string(db_path="research_memory.db")

# BUG 13: Wrong compilation method - should be compile() not build()
app = workflow.build(checkpointer=checkpointer)

def main():
    """Run the research assistant."""
    print("🔬 Advanced Research Assistant Started!")
    print("Ask me to research any topic. I can search, analyze, and summarize.")
    print("Type 'quit' to exit, 'approve' to approve pending requests")
    print("-" * 60)
    
    thread_config = {"configurable": {"thread_id": "research-session-1"}}
    
    while True:
        user_input = input("\nResearcher: ")
        
        if user_input.lower() == 'quit':
            break
            
        if user_input.lower() == 'approve':
            print("✅ Approval granted - continuing research...")
            # BUG 14: Wrong resume mechanism
            app.update_state(thread_config, {"approved_by_human": True})
            continue
        
        # BUG 15: Wrong message type for user input - should be HumanMessage
        user_message = AIMessage(content=user_input)
        
        initial_state = {
            "messages": [user_message],
            "research_query": user_input,
            "research_progress": [],
            "sources_found": [],
            "requires_approval": False,
            "approved_by_human": False,
            "summary": ""
        }
        
        try:
            # BUG 16: Wrong config parameter name - should be config not thread_config
            for event in app.stream(initial_state, thread_config=thread_config):
                for node, value in event.items():
                    if "messages" in value and value["messages"]:
                        last_message = value["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            print(f"Assistant: {last_message.content}")
                    
                    if value.get("requires_approval"):
                        print("⏸️  Waiting for human approval (type 'approve' to continue)")
                        break
                        
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()