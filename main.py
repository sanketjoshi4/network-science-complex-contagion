import matplotlib.pyplot as plt
import networkx as nx
import random as r
import numpy as np


class Contagion:

    def __init__(self, t_type, t_num=None, t_fr=None, rate=1):
        """
        This initializes properties of a complex contagion

        :param t_type: Type of threshold : 'NUMERICAL' / 'FRACTIONAL'
        :param t_num: Numerical threshold value - min. # active neighbors
        :param t_fr: Fractional threshold value - min. # active neighbors
        :param rate: (Unused) Placeholder for # steps of spreading to execute in one call to infect_step
        """
        self.threshold_type = t_type
        self.rate = rate
        self.threshold_numerical = t_num
        self.threshold_fractional = t_fr

class NetworkFactory:
    
    def __init__(self, attr = {}):
        """
        Generate different networks that have given attributes initialized
        
        :param attr: dictionary of attributes to be initialized
        """
        self.attr = attr
    
    def random_network(self, n, p, seed=None, directed=False):
        rg = nx.erdos_renyi_graph(n, p, seed, directed)

        for k,v in self.attr.items():
            nx.set_node_attributes(rg, v, k)
        
        return rg
        
    def scale_free_network(self, n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, create_using=None, seed=None):
        sfg = nx.scale_free_graph(n, alpha, beta, gamma, delta_in, delta_out, create_using, seed)
        for k,v in self.attr.items():
            nx.set_node_attributes(sfg, v, k)
        return sfg
    
    # return a lattic with n nodes, each connected to
    # its k neighbors (k - 1 if k is odd), then rewire
    # the network with probability p
    def lattice(self, n, k, p=0):
        lt = nx.watts_strogatz_graph(n,k,p)
        for k,v in self.attr.items():
            nx.set_node_attributes(lt, v, k)
        return lt
    # load the facebook network
    def facebook_network(self):
        with open('slavko.net') as file:
    
            file.readline();
            file.readline();
            fromNode = 0;
            toNode = 0;
            G=nx.DiGraph()

            edge = file.readline()[:-1];
            while edge:
                fromNode, toNode = [int(x) for x in edge.split(' ')]
                G.add_edge(fromNode, toNode)
                edge = file.readline()[:-1]

            for k,v in self.attr.items():
                nx.set_node_attributes(G, v, k)
        
        return G
    # load the email network
    def email_network(self):
        with open('out.arenas-email') as file:
    
            file.readline();
            fromNode = 0;
            toNode = 0;
            G=nx.DiGraph()

            edge = file.readline()[:-1];
            while edge:
                fromNode, toNode = [int(x) for x in edge.split(' ')]
                G.add_edge(fromNode, toNode)
                edge = file.readline()[:-1]

            for k,v in self.attr.items():
                nx.set_node_attributes(G, v, k)
        
        return G
    

class Network:

    def __init__(self, n_max=None):
        """
        Initializes graph properties. Add any constrained property here

        :param n_max: Max number of nodes
        """
        self.n_max = n_max
        self.G = None  # The networkx graph object
        self.G_infected = None  # Networkx graph object of the infection spread

    def generate(self, graph=None):
        """
        This generates either an empty graph or an input graph
        The input graph needs to be in the standard format
        TODO :  @Sihao to create a NetworkFactory class
                Some parameters would be like network, size, dimensions for lattice, data source, etc
                This class has a function that returns a networkx graph object
                This networks object will be passed in the graph parameter of this function

        :param graph: This input networkx graph object
        :return: The graph object
        """
        self.G = graph if graph is not None else nx.Graph()
        self.G_infected = nx.Graph()
        return self.G

    def copy(self):
        """
        Copy the Network object, making a copy of the underlying graph.
        :return: A Network object.
        """
        rv = Network(self.n_max)
        rv.generate(self.G.copy())
        rv.G_infected = self.G_infected.copy()
        return rv

    def randomly_infect(self, p):
        """
        This activates each node with a probability p
        TODO : @Sanket to generalize this for other initialization mechanisms

        :param p: The probability that a node is active
        """
        for n_id, prop in self.G.nodes.items():
            if r.random() <= p:
                prop['active'] = True
                self.G_infected.add_node(n_id, active=True)
            else:
                prop['active'] = False

    def try_infect_node(self, contagion, n_id):
        """
        This infects a node if threshold is satisfied
        It adds this node to the infected graph, along with edges to all its active neighbors
        TODO : @Anmol to see if this can be optimized

        :param contagion: The contagion object
        :param n_id: The id of the node to check on
        :return:
        """
        active_nbrs = [i for i in self.G[n_id] if self.G.nodes[i]['active']]
        active_nbr_count = len(active_nbrs)
        activates = False
        if contagion.threshold_type == 'NUMERICAL':
            activates = active_nbr_count >= contagion.threshold_numerical
        if contagion.threshold_type == 'FRACTIONAL':
            activates = active_nbr_count / len(self.G[n_id]) >= contagion.threshold_fractional
        if activates:
            self.G.nodes[n_id]['active'] = True
            for nbr in active_nbrs:
                self.G_infected.add_edge(nbr, n_id)
        return activates

    def infect_step(self, contagion):
        """
        This simulates a single step of spread of a contagion and returns the newly infected nodes

        :param contagion: The contagion object
        :return: The newly infected list of nodes ids
        """
        infected = []
        for n_id in list(self.G.nodes):
            if not self.G.nodes[n_id]['active'] and self.try_infect_node(contagion, n_id):
                infected.append(n_id)
        return infected

    def get_infected(self):
        """
        This returns the subset of nodes that are currently infected

        :return: List of node ids
        """
        return [n[0] for n in self.G.nodes.items() if n[1]['active']]

    def infect(self, contagion):
        """
        This simulates the spread of a complex contagion
        Each step infects potential nodes in immediate neighborhood
        Termination is when a particular step fails to infect any new nodes

        TODO : @Anmol Try to see if theres a faster way to find the size of this complex component

        :param contagion:
        :return: The # steps taken to infect the complex component, and the fraction of nodes infected in the whole network
        """
        steps = 0
        while len(self.infect_step(contagion)) > 0:
            steps += 1
        return steps, self.G_infected.number_of_nodes() / self.G.number_of_nodes()

    def get_complex_components(self):
        """
        This returns a list of the complex components of the network at the current stage
        It is equivalent to the list of connected components in the infected graph
        :return: List of lists of node ids
        """
        return list(nx.connected_components(self.G_infected))


class RobustnessTester:
    """
    The test function inputs a network object and a contagion and executes the
    test.
    """

    def __init__(self, attack='rewire'):
        """
        Initialize tester for a given type of attack.

        A rewire attack randomly rewires an edge.
        A random attack randomly deletes a node.
        A targeted attack deletes the node with highest degree.

        :param attack: The type of attack to run. Choices are "rewire", "random",
            or "targeted".
        """
        if attack == 'rewire':
            self._attack = self._rewire
        elif attack == 'random':
            self._attack = self._random
        elif attack == 'targeted':
            self._attack = self._targeted
        else:
            raise ValueError('attack value must be either "rewire", "random", or "targeted"')

        # When rewiring, it is difficult to tell when the graph has reached a
        # truly random state. To capture this, the average clustering of the
        # graph needs to be tracked. Once the graph becomes random its
        # average clustering will oscillate up and down over a small range.
        # When this oscillating behavior is observed (ie. the average clustering
        # starts moving back up), then the graph has reached a random state, and
        # there's no reason to keep rewiring.
        self._average_clustering = float('inf')

    def _rewire(self, network, exclude=None):
        """
        Randomly rewire one edge.

        :param network: A Network object.
        :param exclude: A sequence of nodes that should not be selected to rewire.
        :return: bool indicating if rewiring was successful.
        """
        if exclude is None:
            exclude = list()

        avg_clustering = nx.average_clustering(network.G)
        if avg_clustering > self._average_clustering:
            return False
        else:
            self._average_clustering = avg_clustering

        g = network.G
        # Find two random nodes that share an edge.
        try:
            # Ensure n1 has edges and hasn't been excluded by previous searches.
            n1 = r.choice([n for n in list(g.nodes)
                           if g.adj[n] and n not in exclude])
        except IndexError:
            return False
        n2 = r.choice(list(g.adj[n1]))

        # Find a random node that does not share an edge to n1.
        try:
            # If n1 is connected to every node in the graph, rewiring it isn't
            # possible. Retry and exclude n1.
            n3 = r.choice(list(nx.non_neighbors(g, n1)))
        except IndexError:
            exclude.append(n1)
            return self._rewire(network, exclude)

        # Remove the existing edge from the graph.
        g.remove_edge(n1, n2)

        # Add a new edge to the non-neighbor n3.
        g.add_edge(n1, n3)

        return True

    def _random(self, network):
        """
        Randomly remove a node from the graph.

        :param network: A Network object.
        :return: bool indicating if the attack was successful.
        """
        g = network.G
        try:
            n = r.choice(list(g.nodes))
        except IndexError:
            # No more nodes to remove.
            return False

        g.remove_node(n)

        if len(g.nodes) == 0:
            return False

        return True

    def _targeted(self, network):
        """
        Remove the node with highest degree from the graph.

        :param network: A Network object.
        :return: bool indicating if the attack was successful.
        """
        g = network.G
        try:
            largest, _ = max([(n, e) for n, e in g.degree()], key=lambda x: x[1])
        except ValueError:
            # No more nodes to remove.
            return False

        g.remove_node(largest)

        if len(g.nodes) == 0:
            return False

        return True

    def test(self, network, contagion):
        """
        Test the robustness of a network.

        :param network: A Network object.
        :param contagion: A Contagion object.
        :return: list of tuples containing the step of the attack and the
            distribution of complex component sizes.
        """
        base = network.copy()
        net = base.copy()

        step = 0
        rv = []

        while True:
            net.randomly_infect(p=0.5) # TODO: Convert to seed infections like in paper.
            net.infect(contagion=contagion)
            rv.append((step, [len(n) for n in net.get_complex_components()]))

            if not self._attack(base):
                return rv

            net = base.copy()
            step += 1


def simulate(states, iters, nodes, edge_density, infected, threshold_numeric):
    """
    TODO : @Sanket generalize this for any variable along the x and y axis 
    :param states:
    :param iters:
    :param nodes:
    :param edge_density:
    :param infected:
    :param threshold_numeric:
    :return:
    """

    stats = []
    for i in range(states):
        tot_steps = 0
        tot_success = 0
        print('State {}/{}'.format(i, states))
        for j in range(iters):
            net = Network(nodes)
            cont = Contagion(t_type='NUMERICAL', t_num=threshold_numeric[i])
            net.generate(nx.erdos_renyi_graph(nodes, edge_density[i]))
            net.randomly_infect(p=infected[i])
            step, success = net.infect(contagion=cont)
            tot_steps += step
            tot_success += success
        stats.append((tot_steps / iters, tot_success / iters))

    plt.plot(edge_density, [i[0] for i in stats])
    plt.xlabel('Edge Density')
    plt.ylabel('Propogation Time')
    plt.show()

    plt.plot(edge_density, [i[1] for i in stats])
    plt.xlabel('Edge Density')
    plt.ylabel('Relative Component Size')
    plt.show()


if __name__ == '__main__':
    states = 20
    iters = 5
    nodes = 1000

    edge_density = np.linspace(0.01, 0.03, states)
    infected = np.linspace(0.1, 0.1, states)
    threshold_numeric = np.linspace(6, 6, states)

    simulate(states, iters, nodes, edge_density, infected, threshold_numeric)
