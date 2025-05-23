import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_polygraph(G: nx.Graph):
    # copy original graph
    G = G.copy()

    # Calculate a spring layout where k depends on node size
    node_size = 1000 / np.sqrt(len(G))
    node_radius = np.sqrt(node_size / np.pi) / 72  # roughly from points to inches
    k = 4 * node_radius  # scaling factor (adjust freely)

    # add ghost nodes
    olen = len(G)
    ghost_nodes = np.arange(2) + olen

    estart = 0
    for gns in ghost_nodes[::2]:
        G.add_node(gns, idx=-1)
        G.add_edge(gns, estart)
        estart = gns

    estart = olen - 1
    for gne in ghost_nodes[1::2]:
        G.add_node(gne, idx=-1)
        G.add_edge(gne, estart)
        estart = gne

    G.add_edge(gns, gne)

    pos = nx.circular_layout(G)
    pos = nx.spring_layout(
        G,
        pos=pos,
        iterations=500,
        k=k,
        seed=42,
    )

    # remove the ghost nodes from the graph
    for gn in ghost_nodes:
        G.remove_node(gn)

    # fix every fist and last node

    # pos = nx.spring_layout(G,pos=pos,iterations=50)

    colors = [data["idx"] for _, data in G.nodes(data=True)]
    fig = plt.figure(figsize=(10, 10))

    draw_nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,  # size of the nodes
        node_color=colors,  # color of the nodes
        cmap=plt.cm.viridis,  # color map
    )

    # Set edge color to red
    draw_nodes.set_edgecolor("k")
    nx.draw_networkx_edges(G, pos)
    plt.axis("equal")
    plt.axis("off")
    return fig
