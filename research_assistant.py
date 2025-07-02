"""
Advanced Research Assistant using LangGraph with custom state, memory, and human-in-the-loop.
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
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Custom state fields for research tracking
    research_query: str
    research_progress: List[str]
    sources_found: List[str]
    requires_approval: bool
    approved_by_human: bool
    
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
        result = eval(expression)  # Note: eval is used for simplicity, use with caution
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

@tool
def request_human_approval(reason: str) -> str:
    """Request human approval for a sensitive research topic."""
    return f"Human approval requested for the following reason: {reason}"

# System prompt for the research agent
system_prompt = (
    "You are an advanced research assistant. Your goal is to conduct thorough research on a given topic, "
    "using the available tools to find, analyze, and summarize information. "
    "When you encounter sensitive topics that might require human oversight (e.g., politics, ethics), "
    "you must use the 'request_human_approval' tool to get permission before proceeding. "
    "and provide a comprehensive summary of your findings. Track your progress and sources diligently."
)

# Model and tools setup
llm = ChatAnthropic(model="claude-3-haiku-20240307")
tools = [web_search, document_lookup, calculate_stats, request_human_approval]
research_agent = llm.bind_tools(tools)
execute_tools = ToolNode(tools)

# Agent function
def run_agent(state: ResearchState) -> dict:
    """Run the research agent."""
    messages = state["messages"]
    system = SystemMessage(content=system_prompt)
    response = research_agent.invoke([system] + messages)
    return {"messages": [response]}

# Approval node
def request_approval(state: ResearchState) -> dict:
    """Request human approval and wait for it."""
    if not state.get("approved_by_human"):
        raise NodeInterrupt()
    return {}

# Summarization function
def summarize_research(state: ResearchState) -> dict:
    """Summarize the research findings."""
    summary = (
        "Based on the research, here is a summary of the findings:\n"
        f"Query: {state['research_query']}\n"
        f"Progress: {' -> '.join(state['research_progress'])}\n"
        f"Sources: {', '.join(state['sources_found'])}\n"
    )
    return {"summary": summary}

# Routing function
def route_after_agent(state: ResearchState) -> Literal["tools", "approval", "summarize", "end"]:
    """Inspect the AI message for tool calls and route accordingly."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "request_human_approval":
                return "approval"
        return "tools"
    else:
        return "summarize"

# Define the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("agent", run_agent)
workflow.add_node("tools", execute_tools)
workflow.add_node("approval", request_approval)
workflow.add_node("summarize", summarize_research)

# Add edges
workflow.add_edge(START, "agent")

workflow.add_conditional_edge(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "approval": "approval", 
        "summarize": "summarize",
    }
)

# Add edges back to agent
workflow.add_edge("tools", "agent")
workflow.add_edge("approval", "agent")
workflow.add_edge("summarize", END)

# Setup memory with SQLite
checkpointer = SqliteSaver.from_conn_string("research_memory.db")

app = workflow.build(checkpointer=checkpointer)

def main():
    """Run the research assistant."""
    print("🔬 Advanced Research Assistant Started!")
    print("Ask me to research any topic. I can search, analyze, and summarize.")
    print("Type 'quit' to exit.")
    print("-" * 60)
    
    thread_config = {"configurable": {"thread_id": "research-session-1"}}
    
    while True:
        user_input = input("\nResearcher: ")
        if user_input.lower() == 'quit':
            break

        user_message = HumanMessage(content=user_input)
        initial_input = {"messages": [user_message]}

        try:
            for event in app.stream(initial_input, config=thread_config):
                for key, value in event.items():
                    if key == "agent" and value["messages"]:
                        last_message = value["messages"][-1]
                        if last_message.content:
                            print(f"Assistant: {last_message.content}")

        except NodeInterrupt:
            print("\n⏸️  Sensitive topic detected. Requires human approval to proceed.")
            while True:
                approval_input = input("Type 'approve' to continue or 'reject' to stop: ").lower()
                if approval_input == 'approve':
                    print("✅ Approval granted. Resuming research...")
                    # Resume the graph from the point of interruption
                    app.update_state(thread_config, {"approved_by_human": True})
                    for event in app.stream(None, config=thread_config):
                         for key, value in event.items():
                            if key == "agent" and value["messages"] and value["messages"][-1].content:
                                print(f"Assistant: {value['messages'][-1].content}")
                    break 
                elif approval_input == 'reject':
                    print("❌ Research rejected. Please enter a new query.")
                    break

if __name__ == "__main__":
    main()
