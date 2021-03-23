import json
import math
import multiprocessing
import os
import random
from collections import defaultdict #makes defining the initial condition easier

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import EoN
from scipy.interpolate import make_interp_spline


def rate_function(G, node, status, parameters):
    #This function needs to return the rate at which node changes status.
    #
    threshold = parameters[0]
    if status[node] == 'S':
        if len([nbr for nbr in G.neighbors(node) if status[nbr] == 'I']) >= threshold:
            return 1

    return 0

def transition_choice(G, node, status, parameters):
    #this function needs to return the new status of node.  We already
    #know it is changing status.
    #
    #this function could be more elaborate if there were different
    #possible transitions that could happen.
    infected = parameters[1]
    if status[node] == 'S':
        if rate_function(G, node, status, parameters) == 1:
            infected.add(node)
            return 'I'
        return 'S'
    else:
        return status[node]


def get_influence_set(G, node, status, parameters):
    #this function needs to return any node whose rates might change
    #because ``node`` has just changed status.
    #
    #the only neighbors a node might affect are the susceptible ones.

    return {nbr for nbr in G.neighbors(node) if status[nbr] == 'S'}


def rewire(G):
    edges = G.edges()

    for p in np.linspace(0.0, 1.0, 101):
        rv = G.copy()
        for u, v in random.sample(edges, math.floor(p * len(edges))):
            w = u
            while w != u:
                w = random.choice(G.nodes())

            rv.remove_edge(u, v)
            rv.add_edge(u, w)

        yield p, rv


def random_attack(G):
    nodes = G.nodes()

    for p in np.linspace(0.0, 0.99, 100):
        rv = G.copy()
        for n in random.sample(nodes, math.floor(p * len(nodes))):
            rv.remove_node(n)

        yield p, rv


def targeted_attack(G):
    nodes = sorted(G.nodes(), key=lambda n: nx.degree(G, n), reverse=True)

    for p in np.linspace(0.0, 0.99, 100):
        rv = G.copy()
        for n in nodes[:math.floor(p * len(nodes))]:
            rv.remove_node(n)

        yield p, rv

def run_simulation(G, graph_name, attack_name, attack_step, threshold, size, filename):
    start_nodes = sorted(G.nodes(), key=lambda n: nx.degree(G, n), reverse=True)
    ccs = []

    for n in start_nodes:
        infected = {n} | {nbr for nbr in G.neighbors(n)}
        seed_n = len(infected)

        if any([infected.issubset(cc) for cc in ccs]):
            continue

        parameters = (threshold, infected)

        IC = defaultdict(lambda: 'S')
        for node in infected:
            IC[node] = 'I'

        t, S, I = EoN.Gillespie_complex_contagion(
            G,
            rate_function,
            transition_choice,
            get_influence_set,
            IC,
            return_statuses=('S', 'I'),
            tmax=float('inf'),
            parameters=parameters)

        if len(infected) > seed_n:
            ccs.append(infected)

    with open(filename, 'w') as f:
        json.dump(
            dict(graph=graph_name,
                 attack=attack_name,
                 attack_step=attack_step,
                 threshold=threshold,
                 graph_size=size,
                 components=[len(cc) for cc in ccs]),
            f)


def get_components(graph_name):
    rv = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list))))

    d = 'results_{}'.format(graph_name)
    for filename in os.listdir(d):
        with open(os.path.join(d, filename)) as f:
            res = json.load(f)

        rv[res['threshold']][res['graph_size']][res['attack']][res['attack_step']].append(res['components'])

    return rv


def get_histograms(components, size):
    histos = []
    upper = size + 1
    step = size // 10
    bins = list(range(0, upper, step))
    for c in components:
        histos.append(np.histogram(c, bins=bins))

    return histos


def average_histograms(components, size):
    histos = get_histograms(components, size)

    bins = [h[0] for h in histos]
    mean = np.mean(bins, axis=0)
    std = np.std(bins, axis=0)

    return mean, std, histos[0][1]


def average_largest(components):
    largest = [max(c, default=0) for c in components]
    mean = np.mean(largest)
    std = np.std(largest)

    return mean, std


def plot_components(components, threshold, size, attack, graph_name):
    basename = '{}_{}_{}'.format(graph_name, attack, threshold)
    filename_log = '{}_log.pdf'.format(basename)
    filename = '{}.pdf'.format(basename)

    p_values = sorted(components.keys())
    histos = zip(p_values, [average_histograms(components[p], size) for p in p_values])

    data = defaultdict(list)
    x_axis = []

    for p, (mean, std, bins) in histos:
        x_axis.append(p)
        for i, b in enumerate(bins[:-1]):
            start = b
            if i == len(bins)-2:
                end = bins[i+1]
            else:
                end = bins[i+1] - 1

            key = '{}-{}'.format(start, end)

            data[key].append((mean[i], std[i]))


    fig = plt.figure()
    plt.title('{} - {}'.format(graph_name, attack))

    legend = []
    for k, y in data.items():
        y_values = [_y[0] for _y in y]
        std_dev = [_y[1] for _y in y]

        plt.errorbar(x_axis, y_values, std_dev, capsize=2, errorevery=10)
        legend.append(k)

    plt.xlabel('Percent attack')
    plt.ylabel('Number of components')
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('small')
    plt.legend(legend, title='Component size', prop=fontP) # loc='upper left', 
    plt.savefig(filename)

    plt.yscale('log', nonposy='mask')
    plt.savefig(filename_log)
    plt.close(fig)


def plot_largest_components(components, threshold, attack, graph_name):
    basename = '{}_largest_component_{}_{}'.format(graph_name, attack, threshold)
    filename_log = '{}_log.pdf'.format(basename)
    filename = '{}.pdf'.format(basename)

    p_values = sorted(components.keys())
    largest = zip(p_values, [average_largest(components[p]) for p in p_values])

    x_axis = []
    y_axis = []
    std_dev = []

    for p, (mean, std) in largest:
        x_axis.append(p)
        y_axis.append(mean)
        std_dev.append(std)

    fig = plt.figure()
    plt.title('{} - {}'.format(graph_name, attack))
    plt.errorbar(x_axis, y_axis, std_dev, capsize=2, errorevery=10)
    plt.xlabel('Percent attack')
    plt.ylabel('Largest component')
    plt.savefig(filename)

    plt.yscale('log', nonposy='mask')
    plt.savefig(filename_log)
    plt.close(fig)


def main(graphs, attacks, thresholds, num_procs = 1):
    ps = []
    c = 1
    for i in range(10):
        for graph_name, G in graphs:
            results_dir = 'results_{}'.format(graph_name)
            try:
                os.mkdir(results_dir)
            except FileExistsError:
                pass

            size = len(G.nodes())

            for attack_name in attacks:
                if attack_name == 'rewire':
                    attack = rewire
                elif attack_name == 'targeted':
                    attack = targeted_attack
                elif attack_name == 'random':
                    attack = random_attack
                else:
                    raise ValueError('Unknown attack type: {}'.format(attack_name))

                for threshold in thresholds:
                    for s, g in attack(G):
                        while len(ps) >= num_procs:
                            ps = [p for p in ps if p.join(30) is None and p.is_alive()]

                        filename = os.path.join(results_dir, 'robustness_test_{}'.format(c))
                        p = multiprocessing.Process(target=run_simulation, args=(g, graph_name, attack_name, s, threshold, size, filename))
                        p.start()
                        ps.append(p)
                        c += 1

    for p in ps:
        p.join()

# main()
if __name__ == '__main__':
    size = 1000
    graphs = [('watts_strogatz_7', nx.watts_strogatz_graph(size, 7, 0)),
              ('watts_strogatz_8', nx.watts_strogatz_graph(size, 8, 0)),
              ('barabasi_albert_3', nx.barabasi_albert_graph(size, 3)),
              ('barabasi_albert_4', nx.barabasi_albert_graph(size, 4)),
              ('barabasi_albert_5', nx.barabasi_albert_graph(size, 5))]
    attacks = ['rewire', 'targeted', 'random']
    contagion_thresholds = [2, 3]
    num_procs = 1
    main(graphs, attacks, contagion_thresholds, num_procs)

    for graph_name, _ in graphs:
        data = get_components(graph_name)

        for threshold, sizes in data.items():
            for size, attacks in sizes.items():
                for attack, components in attacks.items():
                    plot_components(components, threshold, size, attack, graph_name)
                    plot_largest_components(components, threshold, attack, graph_name)
