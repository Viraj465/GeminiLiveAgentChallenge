import { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';

interface GraphNode extends d3.SimulationNodeDatum {
    id: string;
    label?: string;
    authors?: string[];
    year?: string | number;
    citation_count?: number;
    research_theme?: string;
    contribution_type?: string;
    problem_statement?: string;
    key_finding?: string;
    methodology?: string;
    limitations?: string[];
    figures_tables?: {label: string, type: string, key_finding: string}[];
}

interface GraphEdge extends d3.SimulationLinkDatum<GraphNode> {
    source: string | GraphNode;
    target: string | GraphNode;
    confidence?: number;
    value?: number;
}

interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
    node_count?: number;
    edge_count?: number;
}

interface CitationGraphProps {
    graphData: GraphData;
}

// Helper to get color by theme
function getThemeColor(theme?: string) {
    const t = (theme || '').toLowerCase();
    if (t.includes('ai') || t.includes('artificial intelligence') || t.includes('learning')) return '#3b82f6'; // Blue
    if (t.includes('cv') || t.includes('vision') || t.includes('image')) return '#10b981'; // Green
    if (t.includes('nlp') || t.includes('language') || t.includes('text')) return '#ec4899'; // Emerald/Pink
    if (t.includes('math') || t.includes('theory')) return '#f59e0b'; // Yellow
    if (t.includes('robotics')) return '#8b5cf6'; // Purple
    return '#6b7280'; // Gray
}

// Shortest path using BFS
function findShortestPath(nodes: GraphNode[], edges: GraphEdge[], startId: string, endId: string): string[] {
    const adjList = new Map<string, string[]>();
    nodes.forEach(n => adjList.set(n.id, []));
    
    edges.forEach(e => {
        const sId = typeof e.source === 'string' ? e.source : (e.source as GraphNode).id;
        const tId = typeof e.target === 'string' ? e.target : (e.target as GraphNode).id;
        
        if (sId && tId) {
            adjList.get(sId)?.push(tId);
            adjList.get(tId)?.push(sId); // treat as undirected for pathfinding to show connection
        }
    });

    const queue: string[] = [startId];
    const visited = new Set<string>([startId]);
    const parent = new Map<string, string>();

    while (queue.length > 0) {
        const curr = queue.shift()!;
        if (curr === endId) break;

        const neighbors = adjList.get(curr) || [];
        for (const n of neighbors) {
            if (!visited.has(n)) {
                visited.add(n);
                parent.set(n, curr);
                queue.push(n);
            }
        }
    }

    if (!parent.has(endId)) return [];

    const path: string[] = [endId];
    let step = endId;
    while (parent.has(step)) {
        step = parent.get(step)!;
        path.push(step);
    }
    return path.reverse();
}

export default function CitationGraph({ graphData }: CitationGraphProps) {
    const svgRef = useRef<SVGSVGElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [minCitations, setMinCitations] = useState(0);
    const [activeNode, setActiveNode] = useState<GraphNode | null>(null);
    
    // Pathfinding state
    const [pathNodes, setPathNodes] = useState<string[]>([]);
    const [shortestPath, setShortestPath] = useState<string[]>([]);

    const maxCitations = useMemo(() => {
        if (!graphData?.nodes) return 100;
        return Math.max(10, ...graphData.nodes.map(n => n.citation_count || 0));
    }, [graphData]);

    useEffect(() => {
        if (!graphData || !graphData.nodes || !graphData.edges || !svgRef.current || !containerRef.current) return;

        const container = containerRef.current;
        const width = container.clientWidth;
        const height = container.clientHeight;

        d3.select(svgRef.current).selectAll('*').remove();

        const svg = d3.select(svgRef.current)
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [0, 0, width, height]);

        // Add defs for arrows
        svg.append("defs").append("marker")
            .attr("id", "arrow")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 25) // pushed out so it doesnt hide under node
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#6b7280")
            .attr("opacity", 0.8);

        svg.append("defs").append("marker")
            .attr("id", "arrow-path")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 25)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#ef4444") // red arrow for path
            .attr("opacity", 1);

        const g = svg.append('g');

        const handleZoom = (e: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
            g.attr('transform', e.transform as unknown as string);
        };
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 4])
            .on('zoom', handleZoom);

        svg.call(zoom);

        // Prepare data
        const nodes: GraphNode[] = graphData.nodes.map(d => Object.assign({}, d));
        const links: GraphEdge[] = graphData.edges.map(d => Object.assign({}, {
            source: d.source,
            target: d.target,
            value: d.confidence || 0.5
        } as GraphEdge));

        const simulation = d3.forceSimulation<GraphNode>(nodes)
            .force('link', d3.forceLink<GraphNode, GraphEdge>(links).id(d => d.id).distance(180))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide().radius(d => {
                const c = (d as GraphNode).citation_count || 0;
                return Math.min(40, 20 + c * 0.5);
            }));

        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('class', 'graph-link')
            .attr('stroke', '#4b5563')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => Math.max(1, (d.value || 0) * 3))
            .attr("marker-end", "url(#arrow)");

        function drag(sim: d3.Simulation<GraphNode, undefined>) {
            function dragstarted(event: d3.D3DragEvent<Element, GraphNode, GraphNode>) {
                if (!event.active) sim.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            function dragged(event: d3.D3DragEvent<Element, GraphNode, GraphNode>) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            function dragended(event: d3.D3DragEvent<Element, GraphNode, GraphNode>) {
                if (!event.active) sim.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            return d3.drag<any, GraphNode>()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended);
        }

        // We use paths to draw circles or squares
        // Circle = Theory, Square = Empirical
        const node = g.append('g')
            .selectAll('path')
            .data(nodes)
            .join('path')
            .attr('class', 'graph-node')
            .attr('d', d => {
                const r = Math.min(30, 12 + (d.citation_count || 0) * 0.5);
                const isTheory = (d.contribution_type || '').toLowerCase().includes('theory') || 
                                 (d.contribution_type || '').toLowerCase().includes('theoretical');
                if (isTheory) {
                    // draw circle
                    return d3.symbol().type(d3.symbolCircle).size(Math.PI * r * r)();
                } else {
                    // draw square
                    return d3.symbol().type(d3.symbolSquare).size(r * r * 4)();
                }
            })
            .attr('fill', d => getThemeColor(d.research_theme))
            .attr('stroke', '#1f2937')
            .attr('stroke-width', 1.5)
            .attr('cursor', 'pointer')
            .call(drag(simulation) as any)
            .on('click', (event, d) => {
                setActiveNode(d);
                
                // Handle Pathfinding selection
                if (event.shiftKey || event.altKey || event.metaKey) {
                    setPathNodes(prev => {
                        const next = [...prev];
                        if (next.includes(d.id)) {
                            return next.filter(id => id !== d.id);
                        }
                        if (next.length >= 2) next.shift();
                        next.push(d.id);
                        return next;
                    });
                }
            });

        // Labels
        const labels = g.append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .attr('class', 'graph-label')
            .attr('dx', d => {
                const r = Math.min(30, 12 + (d.citation_count || 0) * 0.5);
                return r + 8;
            })
            .attr('dy', 5)
            .attr('fill', '#d1d5db')
            .attr('font-size', '12px')
            .attr('font-family', 'sans-serif')
            .attr('pointer-events', 'none')
            .text(d => (d.label || '').substring(0, 50) + ((d.label || '').length > 50 ? '...' : ''));

        // Tooltips (HTML title attribute)
        node.append('title')
            .text(d => `${d.label}\nTheme: ${d.research_theme || 'Unknown'}\nCitations: ${d.citation_count || 0}\nType: ${d.contribution_type || 'Unknown'}`);

        simulation.on('tick', () => {
            link
                .attr('x1', d => (d.source as GraphNode).x || 0)
                .attr('y1', d => (d.source as GraphNode).y || 0)
                .attr('x2', d => (d.target as GraphNode).x || 0)
                .attr('y2', d => (d.target as GraphNode).y || 0);

            node
                .attr('transform', d => `translate(${d.x || 0},${d.y || 0})`);

            labels
                .attr('x', d => d.x || 0)
                .attr('y', d => d.y || 0);
        });

        setTimeout(() => {
            svg.transition().duration(750).call(
                zoom.transform as any,
                d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8).translate(-width / 2, -height / 2)
            );
        }, 100);

        return () => {
            simulation.stop();
        };
    }, [graphData]);

    // Apply Filter & Search Highlighting via D3 directly
    useEffect(() => {
        if (!svgRef.current) return;
        const svg = d3.select(svgRef.current);
        const lowerQuery = searchQuery.toLowerCase();

        svg.selectAll('.graph-node')
            .style('display', (d: any) => (d.citation_count || 0) >= minCitations ? 'block' : 'none')
            .attr('stroke', (d: any) => {
                // Shortest path styling
                if (shortestPath.includes(d.id)) return '#ef4444'; // Red for path
                // Search highlighting
                if (lowerQuery) {
                    const match = 
                        (d.label || '').toLowerCase().includes(lowerQuery) ||
                        (d.key_finding || '').toLowerCase().includes(lowerQuery) ||
                        (d.problem_statement || '').toLowerCase().includes(lowerQuery) ||
                        (d.methodology || '').toLowerCase().includes(lowerQuery);
                    return match ? '#fbbf24' : '#1f2937'; // Amber glow
                }
                // Selected nodes for pathfinding
                if (pathNodes.includes(d.id)) return '#3b82f6';
                
                return '#1f2937';
            })
            .attr('stroke-width', (d: any) => {
                if (shortestPath.includes(d.id)) return 4;
                if (lowerQuery && (
                    (d.label || '').toLowerCase().includes(lowerQuery) ||
                    (d.key_finding || '').toLowerCase().includes(lowerQuery)
                )) return 3;
                if (pathNodes.includes(d.id)) return 3;
                return 1.5;
            })
            .style('filter', (d: any) => {
                if (lowerQuery && (
                    (d.label || '').toLowerCase().includes(lowerQuery) ||
                    (d.key_finding || '').toLowerCase().includes(lowerQuery)
                )) return 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.8))';
                return 'none';
            });

        svg.selectAll('.graph-label')
            .style('display', (d: any) => (d.citation_count || 0) >= minCitations ? 'block' : 'none')
            .style('font-weight', (d: any) => {
                if (shortestPath.includes(d.id)) return 'bold';
                return 'normal';
            })
            .style('fill', (d: any) => {
                if (shortestPath.includes(d.id)) return '#ef4444';
                return '#d1d5db';
            });

        svg.selectAll('.graph-link')
            .style('display', (d: any) => {
                const s = typeof d.source === 'string' ? d.source : d.source.id;
                const t = typeof d.target === 'string' ? d.target : d.target.id;
                const sNode = graphData.nodes.find(n => n.id === s);
                const tNode = graphData.nodes.find(n => n.id === t);
                if ((sNode?.citation_count || 0) < minCitations || (tNode?.citation_count || 0) < minCitations) {
                    return 'none';
                }
                return 'block';
            })
            .attr('stroke', (d: any) => {
                const sId = typeof d.source === 'string' ? d.source : d.source.id;
                const tId = typeof d.target === 'string' ? d.target : d.target.id;
                
                // If link is part of shortest path
                if (shortestPath.includes(sId) && shortestPath.includes(tId)) {
                    // Check if it's adjacent in path
                    const sIdx = shortestPath.indexOf(sId);
                    const tIdx = shortestPath.indexOf(tId);
                    if (Math.abs(sIdx - tIdx) === 1) return '#ef4444';
                }
                return '#4b5563';
            })
            .attr('stroke-width', (d: any) => {
                const sId = typeof d.source === 'string' ? d.source : d.source.id;
                const tId = typeof d.target === 'string' ? d.target : d.target.id;
                if (shortestPath.includes(sId) && shortestPath.includes(tId)) {
                    const sIdx = shortestPath.indexOf(sId);
                    const tIdx = shortestPath.indexOf(tId);
                    if (Math.abs(sIdx - tIdx) === 1) return 3;
                }
                return Math.max(1, (d.value || 0) * 3);
            })
            .attr('marker-end', (d: any) => {
                const sId = typeof d.source === 'string' ? d.source : d.source.id;
                const tId = typeof d.target === 'string' ? d.target : d.target.id;
                if (shortestPath.includes(sId) && shortestPath.includes(tId)) {
                    const sIdx = shortestPath.indexOf(sId);
                    const tIdx = shortestPath.indexOf(tId);
                    if (Math.abs(sIdx - tIdx) === 1) return 'url(#arrow-path)';
                }
                return 'url(#arrow)';
            });

    }, [searchQuery, minCitations, shortestPath, pathNodes, graphData]);

    // Compute shortest path when two nodes are selected
    useEffect(() => {
        if (pathNodes.length === 2 && graphData) {
            const path = findShortestPath(graphData.nodes, graphData.edges, pathNodes[0], pathNodes[1]);
            setShortestPath(path);
        } else {
            setShortestPath([]);
        }
    }, [pathNodes, graphData]);

    if (!graphData) return null;

    return (
        <div className="flex flex-col h-full bg-[#0c0d12] relative">
            {/* Header & Controls */}
            <div className="bg-[#1a1b23] border-b border-white/[0.06] px-4 py-3 flex items-center justify-between z-10 shrink-0 shadow-md">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2 text-indigo-400 font-medium text-sm">
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                        </svg>
                        Citation Network
                    </div>

                    {/* Searchability */}
                    <div className="flex items-center bg-black/30 rounded-md border border-white/[0.05] px-2 py-1 h-8">
                        <svg className="w-4 h-4 text-gray-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                        <input
                            type="text"
                            placeholder="Search papers, concepts..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="bg-transparent text-xs text-white outline-none w-48 placeholder-gray-500"
                        />
                    </div>

                    {/* Filter by Impact */}
                    <div className="flex items-center gap-3">
                        <span className="text-xs text-gray-400">Min Citations: {minCitations}</span>
                        <input
                            type="range"
                            min="0"
                            max={maxCitations}
                            value={minCitations}
                            onChange={(e) => setMinCitations(parseInt(e.target.value))}
                            className="w-24 accent-indigo-500 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                    </div>
                </div>

                <div className="flex items-center gap-4 text-xs">
                    {/* Pathfinding Indicator */}
                    <div className="flex items-center gap-2">
                        {pathNodes.length > 0 && (
                            <button 
                                onClick={() => {setPathNodes([]); setShortestPath([]);}}
                                className="text-gray-400 hover:text-white bg-white/5 px-2 py-1 rounded transition"
                            >
                                Clear Path
                            </button>
                        )}
                        <span className="text-gray-500">
                            Pathfinding: <span className="text-white">{pathNodes.length}/2 selected</span>
                            <span className="ml-1 opacity-60">(Shift+Click)</span>
                        </span>
                    </div>

                    <div className="text-gray-500 font-mono bg-black/20 px-2 py-1 rounded">
                        {graphData.node_count} nodes • {graphData.edge_count} edges
                    </div>
                </div>
            </div>

            {/* D3 Canvas container */}
            <div className="flex-1 flex overflow-hidden relative">
                <div ref={containerRef} className="flex-1 relative overflow-hidden bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-gray-900 via-[#0c0d12] to-black cursor-move">
                    <svg ref={svgRef} className="w-full h-full block" />

                    {/* Overlay legend */}
                    <div className="absolute bottom-4 left-4 text-[10px] text-gray-400 bg-black/60 px-3 py-2 rounded-lg border border-white/[0.05] pointer-events-none flex flex-col gap-1.5 backdrop-blur-sm">
                        <div className="font-semibold text-gray-300 mb-1">Legend</div>
                        <div className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-[#3b82f6]"></div> AI / Learning</div>
                        <div className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-[#10b981]"></div> Computer Vision</div>
                        <div className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-[#ec4899]"></div> NLP / Text</div>
                        <div className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-[#f59e0b]"></div> Math / Theory</div>
                        <div className="mt-1 flex items-center gap-2 border-t border-white/10 pt-1.5">
                            <svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="#6b7280"/></svg> Theory
                        </div>
                        <div className="flex items-center gap-2">
                            <svg width="12" height="12"><rect x="2" y="2" width="8" height="8" fill="#6b7280"/></svg> Empirical
                        </div>
                        <div className="mt-1 pt-1 border-t border-white/10 text-[9px] opacity-70">
                            Size = Citation Count<br/>
                            Shift+Click = Pathfinding
                        </div>
                    </div>
                </div>

                {/* Side Panel for Expanded Node Detail */}
                {activeNode && (
                    <div className="w-80 bg-[#161720] border-l border-white/[0.06] flex flex-col h-full overflow-y-auto shrink-0 animate-in slide-in-from-right-4 duration-200">
                        <div className="p-4 border-b border-white/[0.05] flex justify-between items-start sticky top-0 bg-[#161720]/95 backdrop-blur z-10">
                            <h3 className="font-semibold text-white leading-tight pr-4">
                                {activeNode.label}
                            </h3>
                            <button 
                                onClick={() => setActiveNode(null)}
                                className="text-gray-500 hover:text-white shrink-0 p-1"
                            >
                                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"/></svg>
                            </button>
                        </div>
                        
                        <div className="p-4 flex flex-col gap-4 text-sm">

                            {/* ── Concept / Topic Root node ── */}
                            {activeNode.id === 'topic_root' ? (
                                <div className="flex flex-col gap-3">
                                    <div className="flex flex-wrap gap-2">
                                        <span className="bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded text-xs font-semibold">
                                            Research Topic
                                        </span>
                                        <span className="bg-white/5 border border-white/10 px-2 py-0.5 rounded text-xs text-gray-400">
                                            Central Node
                                        </span>
                                    </div>
                                    <p className="text-gray-400 text-[13px] leading-relaxed">
                                        This is the central research topic node. All papers in the corpus are connected to it. Click any paper node to view its full details, key findings, methodology, and limitations.
                                    </p>
                                    <div className="bg-indigo-500/5 border border-indigo-500/10 rounded p-3 text-[13px] text-indigo-200/80 leading-relaxed">
                                        <span className="font-semibold text-indigo-300">Papers in corpus: </span>
                                        {(graphData.node_count || 1) - 1}
                                    </div>
                                </div>
                            ) : (
                                <>
                                    {/* Metadata — paper nodes only */}
                                    <div className="flex flex-wrap gap-2">
                                        {activeNode.year && (
                                            <span className="bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded text-xs">
                                                {activeNode.year}
                                            </span>
                                        )}
                                        <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded text-xs flex items-center gap-1">
                                            <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/></svg>
                                            {activeNode.citation_count || 0} citations
                                        </span>
                                        {activeNode.contribution_type && (
                                            <span className="bg-white/5 border border-white/10 px-2 py-0.5 rounded text-xs text-gray-400 capitalize">
                                                {activeNode.contribution_type}
                                            </span>
                                        )}
                                        {activeNode.research_theme && (
                                            <span className="bg-white/5 border border-white/10 px-2 py-0.5 rounded text-xs text-gray-400 capitalize">
                                                {activeNode.research_theme}
                                            </span>
                                        )}
                                    </div>

                                    {/* Authors */}
                                    {activeNode.authors && activeNode.authors.length > 0 && (
                                        <div className="text-gray-400 text-xs leading-relaxed">
                                            <span className="text-gray-500 font-semibold uppercase tracking-wider text-[10px]">Authors — </span>
                                            {activeNode.authors.join(', ')}
                                        </div>
                                    )}

                                    {/* Problem Statement */}
                                    {activeNode.problem_statement && (
                                        <div className="space-y-1 mt-1">
                                            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Problem Statement</h4>
                                            <p className="text-gray-400 text-[13px] leading-relaxed">
                                                {activeNode.problem_statement}
                                            </p>
                                        </div>
                                    )}

                                    {/* Key Finding */}
                                    {activeNode.key_finding && (
                                        <div className="space-y-1 mt-1">
                                            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Key Finding</h4>
                                            <p className="text-indigo-200/90 text-[13px] leading-relaxed bg-indigo-500/5 p-2 rounded border border-indigo-500/10">
                                                {activeNode.key_finding}
                                            </p>
                                        </div>
                                    )}

                                    {/* Methodology */}
                                    {activeNode.methodology && (
                                        <div className="space-y-1 mt-1">
                                            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Methodology</h4>
                                            <p className="text-gray-400 text-[13px] leading-relaxed">
                                                {activeNode.methodology}
                                            </p>
                                        </div>
                                    )}

                                    {/* Limitations */}
                                    {activeNode.limitations && activeNode.limitations.filter(Boolean).length > 0 && (
                                        <div className="space-y-1 mt-1">
                                            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Limitations</h4>
                                            <ul className="text-rose-200/80 text-[13px] leading-relaxed list-disc pl-4 space-y-1">
                                                {activeNode.limitations.filter(Boolean).map((lim, i) => (
                                                    <li key={i}>{lim}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}

                                    {/* Figures & Tables */}
                                    {activeNode.figures_tables && activeNode.figures_tables.length > 0 && (
                                        <div className="space-y-1 mt-2 mb-4 border-t border-white/5 pt-4">
                                            <h4 className="text-xs font-semibold text-gray-300 uppercase tracking-wider">Figures & Tables</h4>
                                            <div className="flex flex-col gap-3 mt-2">
                                                {activeNode.figures_tables.map((ft, i) => (
                                                    <div key={i} className="bg-black/20 p-2 rounded border border-white/5">
                                                        <div className="flex items-center gap-2 mb-1">
                                                            <span className="text-[10px] uppercase font-bold text-teal-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                                                                {ft.type}
                                                            </span>
                                                            <span className="text-xs font-medium text-gray-200">{ft.label}</span>
                                                        </div>
                                                        <p className="text-gray-400 text-[12px] leading-snug">
                                                            {ft.key_finding}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Empty state — paper has no details yet */}
                                    {!activeNode.key_finding && !activeNode.problem_statement && !activeNode.methodology && (
                                        <p className="text-gray-500 text-[13px] italic">
                                            No detailed metadata available for this paper. This may be due to extraction limitations.
                                        </p>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
