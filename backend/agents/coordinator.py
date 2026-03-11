from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from agents.searchagent.search_agent import search_papers
from agents.extractionagent.extraction_agent import extract_papers
from agents.synthesisagent.synthesis_agent import synthesize_findings
from agents.citationagent.citation_agent import build_citation_graph
from agents.reportagent.report_agent import generate_report

# Register each sub-agent's main function as an ADK tool
search_tool = FunctionTool(func=search_papers)
extract_tool = FunctionTool(func=extract_papers)
synth_tool = FunctionTool(func=synthesize_findings)
citation_tool = FunctionTool(func=build_citation_graph)
report_tool = FunctionTool(func=generate_report)

coordinator = LlmAgent(
    name="ResearchCoordinator",
    model="gemini-2.5-pro-preview-05-06",
    description="Orchestrates all research sub-agents to complete a literature review",
    instruction="""
    You are a research coordinator. When given a research query:
    1. Use search_papers to find relevant papers on the web
    2. Use extract_papers to extract text from downloaded PDFs
    3. Use synthesize_findings to analyze and cross-reference findings
    4. Use build_citation_graph to map relationships between papers
    5. Use generate_report to write the final literature review
    
    Execute steps in order. Pass outputs from each step as inputs to the next.
    Be thorough but efficient. Target 10-20 papers minimum.
    """,
    tools=[search_tool, extract_tool, synth_tool, citation_tool, report_tool]
)