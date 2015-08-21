__author__ = 'Burtsev'

import networkx as nx
import matplotlib.pyplot as plot
import scipy as mth


def circular_layout_sorted(G, dim=2, scale=1):
    # dim=2 only
    """Position nodes on a circle.

    Parameters
    ----------
    G : NetworkX graph

    dim : int
       Dimension of layout, currently only dim=2 is supported

    scale : float
        Scale factor for positions

    Returns
    -------
    dict :
       A dictionary of positions keyed by node
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("circular_layout() requires numpy: http://scipy.org/ ")
    if len(G)==0:
        return {}
    if len(G)==1:
        return {G.nodes()[0]:(1,)*dim}
    t = np.arange(0,2.0*np.pi,2.0*np.pi/len(G),dtype=np.float32)
    pos = np.transpose(np.array([np.cos(t),np.sin(t)]))
    # pos = nx._rescale_layout(pos,scale=scale)
    sorted_nodes = sorted(G.nodes())
    return {sorted_nodes[i]: pos[i] for i in range(len(pos))}

def drawNet(net):
    """draws the FS network"""

    G = nx.MultiDiGraph()
    G.add_nodes_from(sorted(net.keys()))
    net_activity = []
    for fs in net.keys():
        net_activity.append(net[fs].activity)
        for synapse in net[fs].lateralWeights.keys():
            G.add_edge(synapse, fs, key=2,
                       weight=(net[fs].lateralWeights[synapse]))
        for synapse in net[fs].goalWeights.keys():
            G.add_edge(synapse, fs, key=1,
                       weight=net[fs].goalValues[synapse])
        for synapse in net[fs].problemWeights.keys():
            G.add_edge(synapse, fs, key=0,
                       weight=net[fs].problemValues[synapse])
        for synapse in net[fs].controlWeights.keys():
            G.add_edge(synapse, fs, key=3,
                       weight=abs(net[fs].controlWeights[synapse]))
    node_layout = circular_layout_sorted(G)
    plot.cla()
    nx.draw_networkx_nodes(G, pos=node_layout, node_size=800,
                           node_color=net_activity, cmap=plot.cm.Reds)
    nx.draw_networkx_labels(G, pos=node_layout)
    ar = plot.axes()
    actArrStyle = dict(arrowstyle='fancy',
                       shrinkA=20, shrinkB=20, aa=True,
                       fc="red", ec="none", alpha=0.9, lw=0,
                       connectionstyle="arc3,rad=-0.1", )
    inhibitionArrStyle = dict(arrowstyle='fancy',
                              shrinkA=20, shrinkB=20, aa=True,
                              fc="blue", ec="none", alpha=0.6,
                              connectionstyle="arc3,rad=-0.7", )
    predArrStyle = dict(arrowstyle='fancy',
                        shrinkA=20, shrinkB=20, aa=True,
                        fc="green", ec="none", alpha=0.1,
                        connectionstyle="arc3,rad=0.2", )
    for vertex in G.edges(keys=True, data=True):  # drawing links
        if vertex[3]['weight'] != 0:
            coords = [node_layout[vertex[1]][0],
                      node_layout[vertex[1]][1],
                      node_layout[vertex[0]][0],
                      node_layout[vertex[0]][1]]
            if vertex[2] == 0:
                if vertex[3]['weight'] > 0:
                    actArrStyle['alpha'] = 0.1
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
                inhibitionArrStyle['fc'] = plot.cm.jet((1+vertex[3]['weight'])*128)
                ar.annotate('', (coords[0], coords[1]), (coords[2], coords[3]),
                            arrowprops=inhibitionArrStyle)
            if vertex[2] == 3:
                actArrStyle['alpha'] = 0.1
                actArrStyle['fc'] = plot.cm.RdPu(vertex[3]['weight']*255)
                ar.annotate('', (coords[0], coords[1]), (coords[2], coords[3]),
                            arrowprops=actArrStyle)

    ar.xaxis.set_visible(False)
    ar.yaxis.set_visible(False)
    plot.subplots_adjust(left=0.0, right=1., top=1., bottom=0.0)

    # todo - handle self-links


def drawStateTransitions(net, dim):
    """draws transitions"""

    max_fs = max(net.keys())
    G = nx.MultiDiGraph()
    states = []
    for fs in net.values():
        st = vec2st(dict(fs.problemValues.items()[:dim]))
        if st not in states:
            states.append(st)
        start = vec2st(dict(fs.problemValues.items()[:dim]))
        end = vec2st(dict(fs.goalValues.items()[:dim]))
        G.add_edge(start, end, key=0, weight=fs.ID, label=str(fs.ID))

    # G.add_nodes_from(states)
    node_layout = nx.circular_layout(G)
    plot.cla()
    nx.draw_networkx_nodes(G, pos=node_layout, node_size=800)
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
    print "tansitions:"+str(len(G.edges()))
    print "max_fs:", max_fs
    for vertex in G.edges(keys=True, data=True):  # drawing links
        if vertex[3]['weight'] != 0:
            coords = [node_layout[vertex[1]][0],
                      node_layout[vertex[1]][1],
                      node_layout[vertex[0]][0],
                      node_layout[vertex[0]][1]]
            if vertex[2] == 0:
                if vertex[3]['weight'] > 0:
                    actArrStyle['connectionstyle'] = 'arc3,rad='\
                                                     + str(0.4*mth.log(vertex[3]['weight'])/mth.log(max_fs)+mth.rand()*0.01)
                    print "fs:", vertex[3]['weight'], ' ann:', actArrStyle['connectionstyle']
                    ar.annotate('',
                                (coords[0], coords[1]), (coords[2], coords[3]),
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


def vec2st(d):
    st = ''
    for k in sorted(d.keys()):
        st = st + str(d[k])
    return st
