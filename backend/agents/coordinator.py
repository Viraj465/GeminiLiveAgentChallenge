import google.adk
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from agents.searchagent.search_agent import search_papers
from agents.extractionagent.extraction_agent import extract_papers
from agents.synthesisagent.synthesis_agent import synthesize_findings
from agents.citationagent.citation_agent import build_citation_graph
from agents.reportagent.report_agent import generate_report
from prompts import COORDINATOR_INSTRUCTION
from config import settings

# Register each sub-agent's main function as an ADK tool
search_tool = FunctionTool(func=search_papers)
extract_tool = FunctionTool(func=extract_papers)
synth_tool = FunctionTool(func=synthesize_findings)
citation_tool = FunctionTool(func=build_citation_graph)
report_tool = FunctionTool(func=generate_report)

coordinator = LlmAgent(
    name="ResearchCoordinator",
    model=settings.GOOGLE_REASONING_MODEL,  # Use config instead of hardcoded model
    description="Orchestrates all research sub-agents to complete a literature review",
    instruction=COORDINATOR_INSTRUCTION,
    tools=[search_tool, extract_tool, synth_tool, citation_tool, report_tool]
)
