function draw(graph) {
    var zoom = d3.zoom()
        .scaleExtent([0.05, 2])
        .on("zoom", zoomed);

    var svg = d3.select("svg").call(zoom),
        g = svg.append("g"),
        width = parseFloat(svg.style("width")),
        height = parseFloat(svg.style("height"));

    var transform = d3.zoomIdentity
        .translate(width / 6, height / 6)
        .scale(0.5);

    var color = d3.scaleSequential(d3.interpolateTurbo);

    var simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id((d) => d.id).distance(5))
        .force("charge", d3.forceManyBody().strength(-10))
        .force("center", d3.forceCenter(width / 2, height / 2));

    //svg.call(zoom.transform, transform);
    
    var link = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph.links)
        .enter().append("line")
        .attr("class", "link");

    var minCol = Math.max(1, Math.min.apply(null, graph.nodes.map(n => n.color)));
    var maxCol = Math.max(1, Math.max.apply(null, graph.nodes.map(n => n.color)));
    var maxSize = Math.max(1, Math.max.apply(null, graph.nodes.map(n => n.size)));

    var node = g.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(graph.nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", (d) => Math.sqrt(100.0 * d.size / maxSize))
        .attr("fill", (d) => color((d.color - minCol) / (maxCol - minCol)))
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("title")
        .text((d) => "color: " + d.color.toFixed(3) + "\nnode: " + d.id + "\nsize: " + d.size);

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);

    function ticked() {
        link
            .attr("x1", (d) => d.source.x)
            .attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x)
            .attr("y2", (d) => d.target.y);
        node
            .attr("cx", (d) => d.x)
            .attr("cy", (d) => d.y);
    }

    function dragstarted(event, d) {
        if (!event.active) {
            simulation.alphaTarget(0.3).restart();
        }
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) {
            simulation.alphaTarget(0);
        }
        d.fx = null;
        d.fy = null;
    }

    function zoomed(event) {
        g.attr("transform", event.transform);
    }

}

draw(graph_json);
