"""
Advanced Research Assistant using LangGraph with custom state, memory, and human-in-the-loop.
"""

import os
from typing import Annotated, List, Optional, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

load_dotenv()

# Custom state with research tracking
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Custom state fields for research tracking
    research_query: Optional[str]
    research_progress: Optional[List[str]]
    sources_found: Optional[List[str]]
    requires_approval: Optional[bool]
    approved_by_human: Optional[bool]
    summary: Optional[str]

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
        result = eval(expression)  # Note: eval is unsafe in production
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Error in calculation: {str(e)}"

@tool
def request_human_approval(topic: str) -> str:
    """Request human approval for sensitive research topics."""
    approval_data = interrupt({
        "type": "approval_request",
        "topic": topic,
        "message": f"The topic '{topic}' requires human approval before proceeding with research."
    })
    return f"Human approval {'granted' if approval_data.get('approved') else 'denied'} for topic: {topic}"

# Define tools list
tools = [web_search, document_lookup, calculate_stats, request_human_approval]

# Initialize LLM with tools
llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)

def run_agent(state: ResearchState):
    """Run the research agent with current state."""
    system_prompt = """You are an advanced research assistant. You can:
    1. Search the web for information
    2. Look up internal documents  
    3. Perform calculations and analysis
    4. Request human approval for sensitive topics
    
    Current research progress: {progress}
    Sources found: {sources}
    
    Be thorough and helpful. For sensitive topics like politics, controversies, 
    or personal information, use the request_human_approval tool first.
    """
    
    # Add system message with context
    messages = state["messages"].copy()
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        progress = state.get("research_progress", [])
        sources = state.get("sources_found", [])
        system_msg = SystemMessage(content=system_prompt.format(
            progress=progress, sources=sources
        ))
        messages.insert(0, system_msg)
    
    # Get response from LLM
    response = llm_with_tools.invoke(messages)
    
    # Update state with response and research tracking
    return {
        "messages": [response],
        "research_query": state.get("research_query") or "General research",
        "research_progress": state.get("research_progress", []) + ["Agent response generated"],
    }

def route_after_agent(state: ResearchState) -> Literal["tools", "approval", "summarize"]:
    """Route after agent based on the last message."""
    last_message = state["messages"][-1]
    
    # Handle case where no tool calls are made
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        # Check if we have enough research to summarize
        progress = state.get("research_progress", [])
        if len(progress) > 3:  # Arbitrary threshold for summarization
            return "summarize"
        else:
            return "tools"  # Default to tools if no clear direction
    
    # Check if there are tool calls
    if last_message.tool_calls:
        # Check if any tool call is for human approval
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "request_human_approval":
                return "approval"
        return "tools"
    
    # No tool calls, check if we should summarize
    if last_message.content and ("summary" in last_message.content.lower() or "conclude" in last_message.content.lower()):
        return "summarize"
    
    return "tools"

def request_approval(state: ResearchState):
    """Handle human approval requests - this node processes the approval workflow."""
    # The interrupt happens in the tool, this node just updates state
    return {
        "requires_approval": True,
        "research_progress": state.get("research_progress", []) + ["Approval requested"]
    }

def summarize_research(state: ResearchState):
    """Generate a final research summary."""
    messages = state["messages"]
    
    # Create summary prompt
    summary_prompt = f"""
    Based on the research conversation above, provide a comprehensive summary including:
    1. Main research query: {state.get('research_query', 'Not specified')}
    2. Key findings from sources
    3. Important data points or calculations
    4. Conclusions and recommendations
    
    Research progress: {state.get('research_progress', [])}
    Sources consulted: {state.get('sources_found', [])}
    
    Provide a well-structured summary.
    """
    
    summary_message = HumanMessage(content=summary_prompt)
    summary_response = llm.invoke(messages + [summary_message])
    
    return {
        "messages": [summary_response], 
        "summary": summary_response.content,
        "research_progress": state.get("research_progress", []) + ["Research summarized"]
    }

# Create the workflow
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("agent", run_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("approval", request_approval)
workflow.add_node("summarize", summarize_research)

# Add edges
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
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
checkpointer = SqliteSaver.from_conn_string(":memory:")  # Use in-memory for demo, change to "research_memory.db" for persistence

app = workflow.compile(checkpointer=checkpointer)

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
        # Initialize state with all custom fields
        initial_input = {
            "messages": [user_message],
            "research_query": user_input,
            "research_progress": ["Research started"],
            "sources_found": [],
            "requires_approval": False,
            "approved_by_human": False,
            "summary": None
        }

        # Stream events and handle interrupts properly
        events = app.stream(initial_input, config=thread_config, stream_mode="values")
        
        try:
            for event in events:
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        print(f"Assistant: {last_message.content}")
        except Exception as e:
            if "interrupt" in str(e).lower():
                print("\n⏸️  Sensitive topic detected. Requires human approval to proceed.")
                while True:
                    approval_input = input("Type 'approve' to continue or 'reject' to stop: ").lower()
                    if approval_input == 'approve':
                        print("✅ Approval granted. Resuming research...")
                        # Resume with Command pattern
                        resume_command = Command(resume={"approved": True})
                        try:
                            for event in app.stream(resume_command, config=thread_config, stream_mode="values"):
                                if "messages" in event and event["messages"]:
                                    last_message = event["messages"][-1]
                                    if hasattr(last_message, 'content') and last_message.content:
                                        print(f"Assistant: {last_message.content}")
                        except Exception as resume_error:
                            print(f"Error during resume: {resume_error}")
                        break
                    elif approval_input == 'reject':
                        print("❌ Research rejected. Please enter a new query.")
                        break
                    else:
                        print("Please type 'approve' or 'reject'")
            else:
                print(f"Error occurred: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Research assistant stopped.")
    except Exception as e:
        print(f"Fatal error: {e}")
        print("Make sure you have set ANTHROPIC_API_KEY in your .env file")


