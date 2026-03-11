import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface GraphNode extends d3.SimulationNodeDatum {
    id: string;
    label?: string;
    authors?: string[];
    year?: string | number;
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

export default function CitationGraph({ graphData }: CitationGraphProps) {
    const svgRef = useRef<SVGSVGElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!graphData || !graphData.nodes || !graphData.edges || !svgRef.current || !containerRef.current) return;

        const container = containerRef.current;
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Clear any existing graph
        d3.select(svgRef.current).selectAll('*').remove();

        const svg = d3.select(svgRef.current)
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [0, 0, width, height]);

        const g = svg.append('g');

        // Define zoom behavior
        const handleZoom = (e: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
            g.attr('transform', e.transform as unknown as string);
        };
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 4])
            .on('zoom', handleZoom);

        svg.call(zoom);

        // Prepare data (deep copy to avoid mutating props)
        const nodes: GraphNode[] = graphData.nodes.map(d => Object.assign({}, d));
        const links: GraphEdge[] = graphData.edges.map(d => Object.assign({}, {
            source: d.source,
            target: d.target,
            value: d.confidence || 0.5
        } as GraphEdge));

        // Set up the simulation
        const simulation = d3.forceSimulation<GraphNode>(nodes)
            .force('link', d3.forceLink<GraphNode, GraphEdge>(links).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide().radius(40));

        // Draw links
        const link = g.append('g')
            .attr('stroke', '#4b5563')
            .attr('stroke-opacity', 0.6)
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke-width', d => Math.max(1, (d.value || 0) * 3));

        // Drag functions
        function drag(sim: d3.Simulation<GraphNode, undefined>) {
            function dragstarted(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>) {
                if (!event.active) sim.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

            function dragged(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }

            function dragended(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>) {
                if (!event.active) sim.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }

            return d3.drag<SVGCircleElement, GraphNode>()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended);
        }

        // Draw nodes container
        const node = g.append('g')
            .attr('stroke', '#1f2937')
            .attr('stroke-width', 1.5)
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', 16)
            .attr('fill', '#3b82f6')
            .attr('cursor', 'pointer')
            .call(drag(simulation) as any);

        // Add labels
        const labels = g.append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .attr('dx', 20)
            .attr('dy', 5)
            .attr('fill', '#d1d5db')
            .attr('font-size', '12px')
            .attr('font-family', 'sans-serif')
            .attr('pointer-events', 'none')
            .text(d => (d.label || '').substring(0, 30) + ((d.label || '').length > 30 ? '...' : ''));

        // Add tooltips
        node.append('title')
            .text(d => `${d.label}\nAuthors: ${d.authors?.join(', ')}\nYear: ${d.year || 'Unknown'}`);

        // Update positions on tick
        simulation.on('tick', () => {
            link
                .attr('x1', d => (d.source as GraphNode).x || 0)
                .attr('y1', d => (d.source as GraphNode).y || 0)
                .attr('x2', d => (d.target as GraphNode).x || 0)
                .attr('y2', d => (d.target as GraphNode).y || 0);

            node
                .attr('cx', d => d.x || 0)
                .attr('cy', d => d.y || 0);

            labels
                .attr('x', d => d.x || 0)
                .attr('y', d => d.y || 0);
        });

        // Apply initial slight zoom/pan to fit
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

    if (!graphData) return null;

    return (
        <div className="flex flex-col h-full bg-[#0c0d12]">
            {/* Header */}
            <div className="bg-[#1a1b23] border-b border-white/[0.06] px-4 py-2.5 flex items-center justify-between z-10">
                <div className="flex items-center gap-3">
                    <div className="flex gap-1.5 opacity-50">
                        <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                        <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                        <div className="w-3 h-3 rounded-full bg-[#28c840]" />
                    </div>
                    <div className="flex items-center gap-2 text-indigo-400 font-medium text-sm">
                        <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                        </svg>
                        Citation Network Map
                    </div>
                </div>
                <div className="text-xs text-gray-500 font-mono">
                    {graphData.node_count} nodes • {graphData.edge_count} edges
                </div>
            </div>

            {/* D3 Canvas container */}
            <div ref={containerRef} className="flex-1 relative overflow-hidden bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-gray-900 via-[#0c0d12] to-black cursor-move">
                <svg ref={svgRef} className="w-full h-full block" />

                {/* Overlay text */}
                <div className="absolute bottom-4 left-4 text-xs text-gray-600 bg-black/40 px-3 py-1.5 rounded-lg border border-white/[0.05] pointer-events-none">
                    Scroll to zoom • Drag to pan • Drag nodes to reposition
                </div>
            </div>
        </div>
    );
}
