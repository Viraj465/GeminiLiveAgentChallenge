"""
graph_builder.py — Citation and idea hierarchy network generator.

Transforms a list of extracted papers into a structured D3-compatible JSON
graph. Nodes represent papers (and core ideas), and edges represent how
ideas build upon one another or cite each other.
"""

import logging

logger = logging.getLogger(__name__)


def generate_citation_graph(papers: list[dict], topic: str) -> dict:
    """
    ADK Tool: Generates a D3 JSON graph mapping how given papers relate to each other.
    Takes a list of paper dicts (must have 'title' and 'authors').
    Returns {'nodes': [...], 'links': [...]}
    """
    logger.info(f"Generating citation graph for {len(papers)} papers on '{topic}'...")

    nodes = []
    links = []

    # 1. Central Topic Node (The core idea hierarchy origin)
    nodes.append({
        "id": "topic_root",
        "name": topic.title(),
        "group": 0,
        "type": "concept"
    })

    # 2. Add Paper Nodes
    for i, paper in enumerate(papers):
        paper_id = f"paper_{i}"
        title = paper.get("title", f"Unknown Paper {i}")
        authors = paper.get("authors", "Unknown Authors")
        
        nodes.append({
            "id": paper_id,
            "name": title,
            "authors": authors,
            "group": 1,
            "type": "paper"
        })

        # Base all papers to the central topic (showing they share the core idea)
        links.append({
            "source": "topic_root",
            "target": paper_id,
            "value": 1,
            "relationship": "explores"
        })

    # 3. Simulate inter-paper citations (Hierarchical Idea Flow)
    # In a real app, this parses the "References" section of the PDF text.
    # Here, we create a mock linear progression showing how Paper N builds on Paper N-1
    for i in range(1, len(papers)):
        source_id = f"paper_{i}"     # Newer paper
        target_id = f"paper_{i-1}"   # Cites older paper
        
        links.append({
            "source": source_id,
            "target": target_id,
            "value": 2,
            "relationship": "cites / builds upon"
        })
        
        # Add random cross-citations for a more organic looking network
        if i > 2:
            links.append({
                "source": source_id,
                "target": f"paper_{i-3}",
                "value": 1,
                "relationship": "references methodology"
            })

    result = {
        "nodes": nodes,
        "edges": links,
        "node_count": len(nodes),
        "edge_count": len(links)
    }
    
    logger.info(f"Generated graph with {len(nodes)} nodes and {len(links)} edges.")
    return result
