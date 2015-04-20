__author__ = 'Burtsev'

import networkx as nx
import matplotlib.pyplot as plot


def drawNet(net):
    """draws the FS network"""

    G = nx.MultiDiGraph()
    G.add_nodes_from(net.keys())
    net_activity = []
    for fs in net.keys():
        net_activity.append(net[fs].activity)
        for synapse in net[fs].lateralWeights.keys():
            G.add_edge(synapse, fs, key=2,
                       weight=abs(net[fs].lateralWeights[synapse]))
        for synapse in net[fs].goalWeights.keys():
            G.add_edge(synapse, fs, key=1,
                       weight=net[fs].goalValues[synapse])
        for synapse in net[fs].problemWeights.keys():
            G.add_edge(synapse, fs, key=0,
                       weight=net[fs].problemValues[synapse])
        for synapse in net[fs].controlWeights.keys():
            G.add_edge(synapse, fs, key=3,
                       weight=abs(net[fs].controlWeights[synapse]))

    node_layout = nx.circular_layout(G)  # nx.graphviz_layout(G,prog="neato")
    plot.cla()
    nx.draw_networkx_nodes(G, pos=node_layout, node_size=800,
                           node_color=net_activity, cmap=plot.cm.Reds)
    nx.draw_networkx_labels(G, pos=node_layout)
    ar = plot.axes()
    actArrStyle = dict(arrowstyle='fancy',
                       shrinkA=20, shrinkB=20, aa=True,
                       fc="red", ec="none", alpha=0.85, lw=0,
                       connectionstyle="arc3,rad=-0.1", )
    inhibitionArrStyle = dict(arrowstyle='fancy',
                              shrinkA=20, shrinkB=20, aa=True,
                              fc="blue", ec="none", alpha=0.6,
                              connectionstyle="arc3,rad=-0.13", )
    predArrStyle = dict(arrowstyle='fancy',
                        shrinkA=20, shrinkB=20, aa=True,
                        fc="green", ec="none", alpha=0.7,
                        connectionstyle="arc3,rad=0.2", )
    for vertex in G.edges(keys=True, data=True):  # drawing links
        # print vertex
        if vertex[3]['weight'] != 0:
            coords = [node_layout[vertex[1]][0],
                      node_layout[vertex[1]][1],
                      node_layout[vertex[0]][0],
                      node_layout[vertex[0]][1]]
            if vertex[2] == 0:
                if vertex[3]['weight'] > 0:
                    actArrStyle['fc'] = plot.cm.YlOrRd(vertex[3]['weight']*255)
                    ar.annotate('', (coords[0], coords[1]), (coords[2], coords[3]),
                                arrowprops=actArrStyle)
                else:
                    actArrStyle['fc'] = plot.cm.Greys(abs(vertex[3]['weight'])*255)
                    print '&&& plot:', vertex
                    ar.annotate('', (coords[0], coords[1]), (coords[2], coords[3]),
                                arrowprops=actArrStyle)
            if vertex[2] == 1:
                predArrStyle['fc'] = plot.cm.Greens(vertex[3]['weight']*255)
                ar.annotate('', (coords[0], coords[1]), (coords[2], coords[3]),
                            arrowprops=predArrStyle)
            if vertex[2] == 2:
                inhibitionArrStyle['fc'] = plot.cm.Blues(vertex[3]['weight']*255)
                ar.annotate('', (coords[0], coords[1]), (coords[2], coords[3]),
                            arrowprops=inhibitionArrStyle)
            if vertex[2] == 3:
                actArrStyle['fc'] = plot.cm.RdPu(vertex[3]['weight']*255)
                ar.annotate('', (coords[0], coords[1]), (coords[2], coords[3]),
                            arrowprops=actArrStyle)

    ar.xaxis.set_visible(False)
    ar.yaxis.set_visible(False)
    plot.subplots_adjust(left=0.0, right=1., top=1., bottom=0.0)
    plot.show()
    # todo - handle self-links