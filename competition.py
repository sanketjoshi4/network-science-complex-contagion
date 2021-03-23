import matplotlib.pyplot as plt
import networkx as nx
import random as r
import imageio
import os
import shutil

INACTIVE = -1
COLOR_DEFAULT = (1, 1, 1)
PERMANENT_INFECTION = False
MAX_STEPS = 50
MAX_LOYALTY = 10 ** 6
SIZE = 200

DRAW = False
DEBUG = True
GIF_DURATION = 6

OUT_DIR = 'scenarios'
SCENARIO = 7
CASE = 9


class Contagion:

    def __init__(self, c_id=0, t_type='NUMERICAL', t_num=None, t_fr=None, rate=1, color=COLOR_DEFAULT,
                 permanent=PERMANENT_INFECTION, strength=1, loyalty=0):
        self.contagion_id = c_id
        self.threshold_type = t_type
        self.strength = strength
        self.rate = rate
        self.threshold_numerical = t_num
        self.threshold_fractional = t_fr
        self.color = color
        self.permanent = permanent
        self.loyalty = loyalty if not permanent else MAX_LOYALTY


class Network:

    def __init__(self, n_max=None, permanent_infection=PERMANENT_INFECTION):
        self.n_max = n_max
        self.G = None  # The networkx graph object
        self.G_infected = {}  # Networkx graph object of the infection spread
        self.contagions = {}
        self.permanent_infection = permanent_infection

    def generate(self, graph=None, remove_isolates=True):
        self.G = graph if graph is not None else nx.Graph()
        if remove_isolates:
            isol = [_ for _ in nx.isolates(self.G)]
            self.G.remove_nodes_from(isol)

        for n_id in list(self.G.nodes):
            self.G.nodes[n_id]['state'] = INACTIVE
            self.G.nodes[n_id]['step'] = 0
        return self.G

    def introduce_contagion(self, contagion, chance=0.5, method='HUB_FIRST'):
        self.contagions[contagion.contagion_id] = contagion
        n_id_source = INACTIVE

        if method == 'RANDOM':
            n_id_source = r.choice([n_id for n_id in self.G.nodes if self.G.node[n_id]['state'] == INACTIVE])

        elif method == 'HUB_FIRST':
            nodes_by_degree = sorted([(n, e) for n, e in self.G.degree()], key=lambda x: x[1])
            nodes_by_degree.reverse()
            for n_id, deg in nodes_by_degree:
                if self.G.nodes[n_id]['state'] == INACTIVE:
                    self.G.nodes[n_id]['state'] = contagion.contagion_id
                    n_id_source = n_id
                    break

        self.G_infected[contagion.contagion_id] = nx.Graph()
        self.G_infected[contagion.contagion_id].add_node(n_id_source)

        for nbr_id in self.G[n_id_source]:
            if r.random() < chance and self.G.nodes[nbr_id]['state'] in [INACTIVE, contagion.contagion_id]:
                self.G.nodes[nbr_id]['state'] = contagion.contagion_id
                self.G_infected[contagion.contagion_id].add_edge(n_id_source, nbr_id)

    def try_infect_node(self, n_id, step):
        initial_contagion_id = self.G.nodes[n_id]['state']
        initial_contagion = self.contagions[initial_contagion_id] if initial_contagion_id != INACTIVE else None

        if initial_contagion_id != INACTIVE and initial_contagion.permanent:
            return

        influence = {contagion_id: 0 for contagion_id, contagion in self.contagions.items()}
        influence[INACTIVE] = 0

        nbrs = self.G[n_id]
        nbr_count = len(nbrs)
        for nbr_id in nbrs:
            nbr_state = self.G.nodes[nbr_id]['state']
            influence[nbr_state] += 1 if nbr_state == INACTIVE else self.contagions[nbr_state].strength

        if initial_contagion_id != INACTIVE:
            influence[initial_contagion_id] += initial_contagion.loyalty

        final_contagion_id = initial_contagion_id
        max_influence = 0
        contagion_influence = 0
        for contagion_id, contagion in self.contagions.items():
            if contagion.threshold_type == 'NUMERICAL':
                contagion_influence = influence[contagion_id] / contagion.threshold_numerical
            if contagion.threshold_type == 'FRACTIONAL':
                contagion_influence = (influence[contagion_id] / contagion.threshold_fractional) / nbr_count
            if contagion_influence >= 1:
                if contagion_influence > max_influence or (
                        contagion_influence == max_influence and r.random() < 0.5):
                    final_contagion_id = contagion_id
                    max_influence = contagion_influence

        if final_contagion_id == initial_contagion_id:
            return False

        self.G.nodes[n_id]['state_next'] = final_contagion_id

        for nbr_id in nbrs:
            if self.G.nodes[nbr_id]['state'] == final_contagion_id:
                self.G_infected[final_contagion_id].add_edge(nbr_id, n_id)

        return True

    def infect_step(self, step):
        infects = False
        for n_id in list(self.G.nodes):
            if self.try_infect_node(n_id, step):
                infects = True
        for n_id in list(self.G.nodes):
            if 'state_next' in self.G.node[n_id] and self.G.node[n_id]['state_next'] is not None:
                self.G.node[n_id]['state'] = self.G.node[n_id]['state_next']
                self.G.node[n_id]['state_next'] = None
                self.G.node[n_id]['step'] = step
                # self.draw()
        return infects

    def infect(self):
        step = 0
        self.draw(step)
        current_composition = self.get_contagion_compositions()
        contagion_sizes = []

        while step < MAX_STEPS:
            self.infect_step(step)
            next_composition = self.get_contagion_compositions()
            sizes = self.get_contagion_sizes()
            contagion_sizes.append(sizes)
            if DEBUG:
                print('Steps : {}, Contagion sizes : {}'.format(step, sizes))
            step += 1
            if current_composition == next_composition:
                self.draw(step)
                break
            current_composition = next_composition
            self.draw(step)

        return step, contagion_sizes

    def get_contagion_sizes(self):
        sizes = {contagion_id: 0 for contagion_id, contagion in self.contagions.items()}
        sizes[INACTIVE] = 0
        for n_id in self.G.nodes:
            sizes[self.G.node[n_id]['state']] += 1
        return sizes

    def get_contagion_compositions(self):
        compositions = {contagion_id: [] for contagion_id, contagion in self.contagions.items()}
        compositions[INACTIVE] = []
        for n_id in self.G.nodes:
            compositions[self.G.node[n_id]['state']].append(n_id)
        return {contagion_id: sorted(nodes) for contagion_id, nodes in compositions.items()}

    def draw(self, step, force=False):

        colors = [self.get_color(self.G.node[n_id]) for n_id in self.G.nodes]
        my_pos = nx.spring_layout(self.G, seed=0)
        nx.draw(self.G, pos=my_pos, node_color=colors, with_labels=DEBUG, edgecolors='gray',
                node_size=400 if DEBUG else 50, edge_color='gray')
        plt.savefig('{}/S{}/C{}/{}.png'.format(OUT_DIR, SCENARIO, CASE, step))
        if not DRAW and not force:
            plt.close()
            return

        plt.show()

    def get_color(self, node):
        state = node['state']
        step = node['step'] if 'step' in node else 0
        if state == INACTIVE:
            return COLOR_DEFAULT
        r, g, b = self.contagions[state].color
        return Network.attenuation(r, step), Network.attenuation(g, step), Network.attenuation(b, step)

    @staticmethod
    def attenuation(num, step, max=0.8):
        return num + (max - num) * (1 - 2 ** (-step / 5))


class CompetitionTester:
    def __init__(self):
        pass

    def execute(self, size, type, t_nums, strengths, seed=None, degree=3):
        net = Network(size)
        if type == 'RANDOM':
            net.generate(nx.erdos_renyi_graph(size, degree / size, seed=seed))
        elif type == 'SCALE_FREE':
            net.generate(nx.barabasi_albert_graph(size, degree, seed=seed))
        elif type == 'CLUSTERED':
            net.generate(nx.gaussian_random_partition_graph(100, 20, 5, 0.1, 0.03))
        net.introduce_contagion(Contagion(c_id=0, t_num=t_nums[0], color=(0.8, 0.2, 0.2), strength=strengths[0]),
                                method='RANDOM')
        net.introduce_contagion(Contagion(c_id=1, t_num=t_nums[1], color=(0.2, 0.8, 0.2), strength=strengths[1]),
                                method='RANDOM')
        return net.infect()[1][-1]

    def simulate(self, iters, repetitions, num_nodes=200):
        results = []
        for _ in range(iters):
            print(_)
            seed = r.randint(1, 100)
            sizes_sum = [0, 0, 0]
            t_nums = [int(lerp(1, 1, _ / iters)), int(lerp(1, 1, _ / iters))]
            strengths = [lerp(1, 0, _ / iters), lerp(1, 0, _ / iters)]
            for __ in range(repetitions):
                sizes = self.execute(num_nodes, 'RANDOM', t_nums, strengths, degree=10)
                sizes_sum[0] += sizes[-1]
                sizes_sum[1] += sizes[0]
                sizes_sum[2] += sizes[1]
            sizes_sum[0] /= repetitions * num_nodes
            sizes_sum[1] /= repetitions * num_nodes
            sizes_sum[2] /= repetitions * num_nodes
            results.append(sizes_sum)
        return results

    def plot(self, results):
        plt.plot(results)
        plt.show()

    def plot_composition(self, result, size):

        plt.plot([i[0] / size for i in result], color='red')
        plt.plot([i[1] / size for i in result], color='blue')
        plt.plot([i[-1] / size for i in result], color='gray')
        plt.savefig('{}/S{}/C{}/composition.png'.format(OUT_DIR, SCENARIO, CASE))
        plt.show()

        steps = len(result)
        images = []
        for filename in ['{}/S{}/C{}/{}.png'.format(OUT_DIR, SCENARIO, CASE, step) for step in range(steps + 1)]:
            images.append(imageio.imread(filename))
            os.remove(filename)

        imageio.mimsave('{}/S{}/C{}/composition.gif'.format(OUT_DIR, SCENARIO, CASE), images,
                        duration=GIF_DURATION / steps)

    def avg_composition(self, results):
        max_steps = max([len(result) for result in results])
        net_result = []
        for step in range(max_steps):
            sum_result_step = [0, 0, 0]
            num_result_step = 0
            for result in results:
                if step < len(result):
                    ord = result[step][0] > result[step][1]
                    sum_result_step[0] += result[step][0] if ord else result[step][1]
                    sum_result_step[1] += result[step][1] if ord else result[step][0]
                    sum_result_step[2] += result[step][-1]
                    num_result_step += 1
            sum_result_step[0] /= num_result_step
            sum_result_step[1] /= num_result_step
            sum_result_step[-1] /= num_result_step
            net_result.append(sum_result_step)
        return net_result

    def scenario(self, size):

        if not os.path.exists('{}/S{}'.format(OUT_DIR, SCENARIO)):
            os.mkdir('{}/S{}'.format(OUT_DIR, SCENARIO))

        if os.path.exists('{}/S{}/C{}'.format(OUT_DIR, SCENARIO, CASE)):
            shutil.rmtree('{}/S{}/C{}'.format(OUT_DIR, SCENARIO, CASE))

        os.mkdir('{}/S{}/C{}'.format(OUT_DIR, SCENARIO, CASE))

        if SCENARIO == 1:
            # random graph, random intro
            net = Network(size, permanent_infection=False)
            net.generate(nx.erdos_renyi_graph(size, 10 / size))
            net.introduce_contagion(chance=1, contagion=Contagion(c_id=0, t_num=3, color=(0.8, 0.2, 0.2)))
            net.introduce_contagion(chance=1, contagion=Contagion(c_id=1, t_num=3, color=(0.2, 0.2, 0.8)))
            result = net.infect()[1]
            # net.draw(force=True)
            self.plot_composition(result, size)

            return result

        if SCENARIO == 2:
            net = Network(size, permanent_infection=False)
            net.generate(nx.erdos_renyi_graph(size, 20 / size))
            net.introduce_contagion(contagion=Contagion(c_id=0, t_num=5, color=(0.8, 0.2, 0.2)))
            net.introduce_contagion(contagion=Contagion(c_id=1, t_num=5, color=(0.2, 0.2, 0.8)))
            result = net.infect()[1]
            self.plot_composition(result, size)
            return result

        if SCENARIO == 3:
            net = Network(size, permanent_infection=False)
            net.generate(nx.barabasi_albert_graph(size, 5))
            net.introduce_contagion(contagion=Contagion(c_id=0, t_num=5, color=(0.8, 0.2, 0.2)))
            net.introduce_contagion(contagion=Contagion(c_id=1, t_num=5, color=(0.2, 0.2, 0.8)))
            result = net.infect()[1]
            self.plot_composition(result, size)
            return result

        if SCENARIO == 4:
            net = Network(size, permanent_infection=False)
            net.generate(nx.karate_club_graph())
            net.introduce_contagion(contagion=Contagion(c_id=0, t_num=3, color=(0.8, 0.2, 0.2)), chance=0.3)
            net.introduce_contagion(contagion=Contagion(c_id=1, t_num=3, color=(0.2, 0.2, 0.8)), chance=0.3)
            result = net.infect()[1]
            self.plot_composition(result, size)
            return result

        if SCENARIO == 5:
            net = Network(size, permanent_infection=False)
            net.generate(nx.relaxed_caveman_graph(5, 10, 0.4))
            net.introduce_contagion(contagion=Contagion(c_id=0, t_num=4, color=(0.8, 0.2, 0.2)), chance=0.5)
            net.introduce_contagion(contagion=Contagion(c_id=1, t_num=4, color=(0.2, 0.2, 0.8)), chance=0.5)
            result = net.infect()[1]
            self.plot_composition(result, size)
            return result

        if SCENARIO == 6:
            net = Network(size, permanent_infection=False)
            net.generate(nx.relaxed_caveman_graph(5, 10, 0.2))
            net.introduce_contagion(contagion=Contagion(c_id=0, t_num=2, color=(0.8, 0.2, 0.2), loyalty=0), chance=0.5)
            net.introduce_contagion(contagion=Contagion(c_id=1, t_num=2, color=(0.2, 0.2, 0.8), loyalty=0), chance=0.5)
            result = net.infect()[1]
            self.plot_composition(result, size)
            return result

        if SCENARIO == 7:
            net = Network(size, permanent_infection=False)
            net.generate(nx.relaxed_caveman_graph(5, 10, 0.2))
            net.introduce_contagion(contagion=Contagion(c_id=0, t_num=2, color=(0.8, 0.2, 0.2), loyalty=0), chance=0.5)
            net.introduce_contagion(contagion=Contagion(c_id=1, t_num=3, color=(0.2, 0.2, 0.8), loyalty=MAX_LOYALTY), chance=0.5)
            result = net.infect()[1]
            self.plot_composition(result, size)
            return result

        return None

    def scenario_final(self, size):
        net = Network(size, permanent_infection=False)
        net.generate(nx.erdos_renyi_graph(size, 10 / size, seed=0))
        net.introduce_contagion(method='RANDOM', chance=0.5,
                                contagion=Contagion(c_id=0, t_num=3, color=(0.8, 0.2, 0.2), loyalty=1))
        net.introduce_contagion(method='RANDOM', chance=0.5,
                                contagion=Contagion(c_id=1, t_num=3, color=(0.2, 0.2, 0.8), loyalty=1))
        result = net.infect()[1]
        # net.draw(force=True)
        self.plot_composition(result, size)
        return result


def lerp(min, range, ratio):
    return min + range * ratio


if __name__ == '__main__':
    print('Begin...')
    CompetitionTester().scenario(SIZE)
    print('End...')
    # ct.plot(ct.simulate(20, 5, num_nodes=200))
