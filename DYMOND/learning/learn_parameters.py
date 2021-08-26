import os
import sys
import pickle
import logging
import errno
import igraph
from scipy.special import comb
from scipy.stats import expon
from ..helpers.gzutils import *


def learn_node_arrival_rates(g, timesteps, tmp_files_dir):
    """
    Estimate node arrival rates

    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :return: node arrival rate
    :rtype: dict
    """
    logging.info('Learning node arrival rates...')
    node_arrival_rates_file = os.path.join(tmp_files_dir, f'node_arrival_rates.pkl')

    if not file_exists(node_arrival_rates_file):
        node_arrivals = {v['nid']: timesteps.index(v['active'])
                         for v in g.vs if v['active']}
        x = list(node_arrivals.values())

        loc, scale = expon.fit(x)
        node_arrival_rate = {'scale': scale, 'loc': loc}
        pickle_save(node_arrival_rates_file, node_arrival_rate)
    else:
        node_arrival_rate = pickle_load(node_arrival_rates_file)

    return node_arrival_rate


def get_active_nodes(g, timesteps, tmp_files_dir):
    """
    Get active nodes in each timestep

    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :return: active nodes per timestep
    :rtype: dict
    """
    active_nodes_file = os.path.join(tmp_files_dir + '/..', f'active_nodes.msg')
    logging.info('Get active nodes (per timestep)')

    if not file_exists(active_nodes_file):
        V = {t: [] for t in timesteps}
        active_nodes = []
        N = g.vcount()

        for idx, v in enumerate(g.vs):
            if idx+1 < N:
                sys.stdout.write(f'{idx+1}, ')
            else:
                sys.stdout.write(f'{idx+1}\n')

            if v['active']:
                active_nodes.append(v)
                for t in timesteps:
                    if t >= v['active']:
                        V[t].append(v.index)

        assert len(active_nodes) > 0

        msgpack_save(active_nodes_file, V)
        logging.info('Saved temp file')
    else:
        V = msgpack_load(active_nodes_file)
        logging.info('Read saved file')

    return V


def get_motifs_graph(g, timesteps, tmp_files_dir, nodes=None):
    """
    Get the motifs in the input graph

    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :param nodes: (optional) nodes to get motifs for
    :type nodes: list
    :return: motifs and motif types
    :rtype: dict
    """
    motifs_file = os.path.join(tmp_files_dir + '/..', f'motifs_in_graph.msg')
    types_file = os.path.join(tmp_files_dir + '/..', f'motif_types_in_graph.msg')

    logging.info('Get motifs in graph')
    if not file_exists(motifs_file) or not file_exists(types_file):
        if not nodes:
            nodes = get_active_nodes(g, timesteps, tmp_files_dir)

        time_code = log_time()

        motifs = set([])
        motif_types = dict()

        for idx_t, t in enumerate(timesteps):
            # Get G_t
            g_t = g.es.select(timestep_eq=t).subgraph(delete_vertices=False)
            g_t.to_undirected()
            g_t.simplify(multiple=True, loops=True)

            for edge in g_t.es:
                u = edge.source
                v = edge.target

                for w in nodes[t]:
                    if w != u and w != v:
                        m = frozenset((u, v, w))
                        edges_used = set([frozenset((e.source, e.target))
                                          for e in g_t.es.select(_within=[u, v, w])])
                        i = len(edges_used)

                        if m not in motifs:
                            motifs.add(m)
                            motif_types[m] = i
                        elif i > motif_types[m]:
                            motif_types[m] = i

        msgpack_save(motifs_file, motifs)
        msgpack_save(types_file, motif_types)
        total_time = time_code('')
        logging.info(f'Finished calculating motifs! {total_time}')

    else:
        logging.info('Read tmp files')
        motifs = msgpack_load(motifs_file)
        motif_types = msgpack_load(types_file)

    return {'motifs': motifs, 'types': motif_types}


def calc_motifs_timesteps(motif_types, g, timesteps, tmp_files_dir):
    """
    Calculate timesteps each motif appears in

    :param motif_types: motif types
    :type motif_types: dict
    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    """
    motifs_timesteps_file = os.path.join(tmp_files_dir + '/..', f'motifs_timesteps.msg')
    logging.info('Calculate timesteps each motif appears in')

    if not file_exists(motifs_timesteps_file):
        time_code = log_time()
        motifs_timesteps = {'timesteps': timesteps,
                            'files': {t: None for t in timesteps}}

        V = get_active_nodes(g, timesteps, tmp_files_dir)

        for idx_t, t in enumerate(timesteps):
            motifs_t_file = os.path.join(tmp_files_dir + '/..', f'motifs_{t}.msg')

            if not file_exists(motifs_t_file):
                # Get G_t
                g_t = g.es.select(timestep_eq=t).subgraph(delete_vertices=False)
                g_t.to_undirected()
                g_t.simplify(multiple=True, loops=True)

                motif_census_t = {1: set([]), 2: set([]), 3: set([])}

                for edge in g_t.es:
                    u = edge.source
                    v = edge.target

                    for w in V[t]:
                        if w != u and w != v:
                            motif = frozenset((u, v, w))
                            i = motif_types[motif]
                            edges_used = set([frozenset((e.source, e.target))
                                              for e in g_t.es.select(_within=[u, v, w])])
                            if len(edges_used) == i:
                                motif_census_t[i].add(motif)

                cnt_1 = len(motif_census_t[1])
                cnt_2 = len(motif_census_t[2])
                cnt_3 = len(motif_census_t[3])

                motifs_t = {'counts': {0: int(comb(len(V[t]), 3, repetition=False)) - cnt_1 - cnt_2 - cnt_3,
                                       1: cnt_1,
                                       2: cnt_2,
                                       3: cnt_3},
                            'motifs': motif_census_t,
                            'V_t': V[t]}

                msgpack_save(motifs_t_file, motifs_t)

            motifs_timesteps['files'][t] = motifs_t_file

        msgpack_save(motifs_timesteps_file, motifs_timesteps)
        total_time = time_code('')
        logging.info(f'Finished calculating motifs per timestep! {total_time}')


def get_motifs_t(t, tmp_files_dir):
    """
    Get motifs in timestep t

    :param t: timestep
    :type t: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :return: motifs at time t
    :rtype: dict
    """
    motifs_t_file = os.path.join(tmp_files_dir + '/..', f'motifs_{t}.msg')
    motifs_t = msgpack_load(motifs_t_file)
    return motifs_t


def learn_motif_proportions(motifs, motif_types, size_V, tmp_files_dir):
    """
    Estimate proportions of each motif type

    :param motifs: motifs
    :type motifs: list
    :param motif_types: motif types
    :type motif_types: dict
    :param size_V: number of nodes
    :type size_V: int
    :param tmp_files_dir:
    :return: motif type proportions
    :rtype: dict
    """
    logging.info('Learn motif proportions')
    motif_proportions_file = os.path.join(tmp_files_dir, 'motif_proportions.msg')
    if not file_exists(motif_proportions_file):
        time_code = log_time()
        possible = int(comb(size_V, 3))
        num_motifs = {i: 0 for i in range(4)}
        for m in motifs:
            i = motif_types[m]
            num_motifs[i] += 1
        num_motifs[0] = possible - sum([num_motifs[i] for i in range(1,4)])
        proportions = {i: num_motifs[i]/possible for i in range(4)}
        total_time = time_code()
        logging.info('Finished motif proportions!')
        msgpack_save(motif_proportions_file, proportions)
    else:
        proportions = msgpack_load(motif_proportions_file)
    return proportions


def get_motif_counts(motifs, motif_types, g, timesteps, tmp_files_dir):
    """
    Estimate motif edge-weighted counts

    :param motifs: motifs
    :type motifs: list
    :param motif_types: motif types
    :type motif_types: dict
    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :return: motif edge-weighted counts and num. of timesteps
    """
    motif_counts_file = os.path.join(tmp_files_dir, f'motif_counts.msg')
    logging.info('Get motif edge-weighted counts')

    if not file_exists(motif_counts_file):
        time_code = log_time()
        motif_counts = {m: {'counts': 0,
                            'timesteps': 0}
                        for m in motifs}

        calc_motifs_timesteps(motif_types, g, timesteps, tmp_files_dir)

        for idx_t, t in enumerate(timesteps):
            motifs_t = get_motifs_t(t, tmp_files_dir)
            motif_counts_t_file = os.path.join(tmp_files_dir, f'motif_counts_{t}.msg')
            if not file_exists(motif_counts_t_file):
                motif_counts_t = {}

                # Get G_t
                g_t = g.es.select(timestep_eq=t).subgraph(delete_vertices=False)
                g_t.to_undirected()
                g_t.simplify(combine_edges=dict(weight="sum"))  # v2

                V_t = set(motifs_t['V_t'])

                for i in [3,2,1]:
                    for m in motifs_t['motifs'][i]:
                        u, v, w = m

                        weight = 0
                        edges_motif = [(e.source, e.target, e['weight'])
                                       for e in g_t.es.select(_within=[u, v, w])]
                        n = len(edges_motif)

                        for (s, d, edge_count) in edges_motif:
                            s_neighbors = set(g_t.neighbors(s))
                            d_neighbors = set(g_t.neighbors(d))

                            num_triangles = len(s_neighbors & d_neighbors)
                            rem_triangles = max([0, edge_count - num_triangles])

                            num_wedges = len(s_neighbors ^ d_neighbors) - 2
                            rem_wedges = max([0, edge_count - num_wedges])

                            num_1edges = len(V_t - s_neighbors - d_neighbors) - 2

                            if i == 3:  # triangle
                                if num_triangles > 0:
                                    weight += min([edge_count, num_triangles]) / num_triangles
                            elif i == 2:  # wedge
                                if num_wedges > 0:
                                    weight += min([rem_triangles, num_wedges]) / num_wedges
                            elif i == 1:  # 1-edge
                                if num_1edges > 0:
                                    weight += rem_wedges / num_1edges

                            motif_cnt = weight / i
                            motif_counts_t[m] = motif_cnt
                            motif_counts[m]['counts'] += motif_cnt
                            motif_counts[m]['timesteps'] += 1

                msgpack_save(motif_counts_t_file, motif_counts_t)
            else:
                motif_counts_t = msgpack_load(motif_counts_t_file)
                for m in motifs:
                    i = motif_types[m]
                    if m in motifs_t['motifs'][i]:
                        motif_counts[m]['counts'] += motif_counts_t[m]
                        motif_counts[m]['timesteps'] += 1

        total_time = time_code('')
        logging.info(f'Finished calculating motif edge-weighted counts! {total_time}')
        msgpack_save(motif_counts_file, motif_counts)
    else:
        logging.info('Read save file')
        motif_counts = msgpack_load(motif_counts_file)

    return motif_counts


def learn_motif_interarrival_rates(motifs, motif_types, g, timesteps, tmp_files_dir):
    """
    Estimate inter-arrival rates per motif type

    :param motifs: motifs
    :type motifs: list
    :param motif_types: motif types
    :type motif_types: dict
    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :return: motif interarrival rates
    :rtype: dict
    """
    logging.info('Learn motif inter-arrival rates')
    motif_interarrivals_file = os.path.join(tmp_files_dir, f'motif_interarrivals.msg')
    if not file_exists(motif_interarrivals_file):
        # individual inter-arrival rates
        num_timesteps = len(timesteps)
        motif_counts = get_motif_counts(motifs, motif_types, g, timesteps, tmp_files_dir)

        logging.info('Estimate individual inter-arrival rates')
        motif_interarrival_rates = {m: motif_counts[m]['counts']/num_timesteps for m in motifs}
        del motif_counts  # free up memory

        # distribution of rates per motif type
        m_types = [3, 2, 1]
        rates_distribution = {i: [motif_interarrival_rates[m]
                                  for m in motifs
                                  if motif_types[m] == i]
                              for i in m_types}

        rates_per_type = {}
        for i in m_types:
            if rates_distribution[i]:
                logging.info(f'Estimate rate for type i={i}')
                x = rates_distribution[i]
                loc, scale = expon.fit(x)
                rates_per_type[i] = {'scale': scale, 'loc': loc}
            else:
                logging.warning(f'Could not calculate rate for type i={i}')
                rates_per_type[i] = {'scale': 0, 'loc': 0}

        del rates_distribution  # free up memory

        logging.info('Save tmp file')
        msgpack_save(motif_interarrivals_file, rates_per_type)
    else:
        logging.info('Read saved file')
        rates_per_type = msgpack_load(motif_interarrivals_file)
    return rates_per_type


def get_node_role_counts(g, timesteps, tmp_files_dir):
    """
    Estimate node role counts

    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :return: node role counts, motif type counts
    :rtype: dict, dict
    """
    logging.info('Estimate node role counts')
    role_counts_file = os.path.join(tmp_files_dir, f'node_role_counts.msg')  # node roles
    type_counts_file = os.path.join(tmp_files_dir, f'node_type_counts.msg')  # node motif types

    if not file_exists(role_counts_file) and not file_exists(type_counts_file):
        time_code = log_time()

        active_nodes = get_active_nodes(g, timesteps, tmp_files_dir)

        roles = ['equal3', 'hub', 'spoke', 'equal2', 'outlier']
        role_counts = {v.index: {r: 0 for r in roles}
                       for v in g.vs}
        type_counts = {v.index: {m_type: 0 for m_type in [3, 2, 1]}
                       for v in g.vs}

        for idx_t, t in enumerate(timesteps):

            nodes_t = set(active_nodes[t])

            # Get G_t
            g_t = g.es.select(timestep_eq=t).subgraph(delete_vertices=False)
            g_t.to_undirected()
            g_t.simplify(combine_edges=dict(weight="sum"))

            for edge in g_t.es:
                u = edge.source
                v = edge.target
                edge_count = edge['weight']

                u_neighbors = set(g_t.neighbors(u))
                v_neighbors = set(g_t.neighbors(v))

                # triangles
                w_triangles = u_neighbors & v_neighbors
                num_triangles = len(w_triangles)
                edge_count_triangle = min([edge_count, num_triangles])
                rem_triangles = max([0, edge_count - num_triangles])

                # wedges
                w_wedges = u_neighbors ^ v_neighbors
                num_wedges = len(w_wedges)
                edge_count_wedge = min([rem_triangles, num_wedges])
                rem_wedges = max([0, rem_triangles - num_wedges])

                # 1-edges
                w_1edges = nodes_t - w_triangles - w_wedges - {u, v}
                num_1edges = len(w_1edges)
                edge_count_1edge = min([rem_wedges, num_1edges])

                if num_triangles > 0:
                    weight_triangle = edge_count_triangle / 2  # equal3 counted twice per triangle
                    for node in [u,v]:
                        role_counts[node]['equal3'] += weight_triangle
                        type_counts[node][3] += weight_triangle

                if rem_triangles > 0 and num_wedges > 0:
                    weight_wedge = edge_count_wedge / num_wedges  # spoke counted once (total) per wedge
                    for w in w_wedges:
                        if w in u_neighbors:
                            hub = u
                            spoke = v
                        elif w in v_neighbors:
                            hub = v
                            spoke = u
                        role_counts[hub]['hub'] += weight_wedge / 2  # hub counted twice per wedge
                        type_counts[hub][2] += weight_wedge / 2
                        role_counts[spoke]['spoke'] += weight_wedge
                        type_counts[spoke][2] += weight_wedge

                if rem_wedges > 0 and num_1edges > 0:
                    weight_1edge = edge_count_1edge / num_1edges
                    for node in [u,v]:
                        role_counts[node]['equal2'] += edge_count_1edge  # equal2 sums to edge count
                        type_counts[node][1] += edge_count_1edge
                    for w in w_1edges:
                        role_counts[w]['outlier'] += weight_1edge  # should be fraction
                        type_counts[w][1] += weight_1edge

        msgpack_save(role_counts_file, role_counts)
        msgpack_save(type_counts_file, type_counts)

        total_time = time_code()
        logging.info(f'Finished node role counts! {total_time}')
    else:
        print('> get node role counts')
        role_counts = msgpack_load(role_counts_file)
        type_counts = msgpack_load(type_counts_file)
    return role_counts, type_counts


def learn_node_roles_distribution(g, timesteps, tmp_files_dir):
    """
    Estimate node role probabilities

    :param g: input graph
    :type g: igraph.Graph
    :param timesteps: graph timesteps
    :type timesteps: int
    :param tmp_files_dir: directory for tmp files
    :type tmp_files_dir: str
    :return: node role probabilities, role counts, motif type counts
    :rtype: dict, dict, dict
    """
    logging.info('Learn node roles distribution')
    node_roles_distr_file = os.path.join(tmp_files_dir, f'node_roles_distr_data.msg')

    if not file_exists(node_roles_distr_file):
        time_code = log_time()
        role_counts, type_counts = get_node_role_counts(g, timesteps, tmp_files_dir)
        roles = ['equal3', 'hub', 'spoke', 'equal2', 'outlier']
        role_distr = {v.index: None for v in g.vs}

        nonzero = 0
        zero = 0
        for vertex in g.vs:
            v = vertex.index
            total = sum([role_counts[v][r] for r in roles])
            if total > 0:
                nonzero += 1
            else:
                zero += 1
            role_distr[v] = {r: role_counts[v][r]/total if total > 0 else 0 for r in roles}

        assert nonzero != 0
        total_time = time_code()
        logging.info(f'Finished node roles distribution! {total_time}')
        msgpack_save(node_roles_distr_file, (role_distr, role_counts, type_counts))
    else:
        logging.info('Read saved file')
        role_distr, role_counts, type_counts = msgpack_load(node_roles_distr_file)
    return role_distr, role_counts, type_counts


def learn_parameters(dataset_dir, dataset_info, g):
    """
    Learn parameters from input graph

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :param dataset_info: dataset information
    :type dataset_info: dict
    :param g: dataset graph
    :type g: igraph.Graph
    """
    time_elapsed = log_time()

    # Path for parameters and tmp files
    params_dir, tmp_files_dir = get_directories_parameters(dataset_dir)
    # File to save model parameters
    model_params_file = os.path.join(params_dir, 'model_params.msg')

    # Learn parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
    model_params = dict()
    model_params['N'] = dataset_info['N']  # number of nodes (i.e., |V|)

    # Learn node arrival rates
    model_params['node arrivals'] = learn_node_arrival_rates(g, dataset_info['timesteps'], tmp_files_dir)

    # Get motifs in graph
    motifs_graph = get_motifs_graph(g,
                                    dataset_info['timesteps'],
                                    tmp_files_dir)

    # Learn motif type inter-arrival rates
    model_params['motif interarrivals'] = learn_motif_interarrival_rates(motifs_graph['motifs'],
                                                                         motifs_graph['types'],
                                                                         g,
                                                                         dataset_info['timesteps'],
                                                                         tmp_files_dir)

    # Learn motif proportions
    model_params['motif proportions'] = learn_motif_proportions(motifs_graph['motifs'],
                                                                motifs_graph['types'],
                                                                dataset_info['N'],
                                                                tmp_files_dir)

    del motifs_graph  # free up memory

    # Learn node roles distribution
    roles_distr, roles_counts, type_counts = learn_node_roles_distribution(g,
                                                                           dataset_info['timesteps'],
                                                                           tmp_files_dir)
    model_params['node roles distr'] = roles_distr
    model_params['node roles counts'] = roles_counts
    model_params['node motif types counts'] = type_counts

    # Save parameters
    msgpack_save(model_params_file, model_params)

    total_time = time_elapsed('')
    logging.info(f'Learn parameters done. {total_time}')


def get_directories_parameters(dataset_dir):
    """
    Get the directories for parameters and temp save files.
    If they don't exist, create the directories.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :return: parameters directory, tmp files directory
    :rtype: str, str
    """
    # Directory to save parameters
    params_dir = os.path.join(dataset_dir, f'learned_parameters')
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    # Directory to save temp files
    tmp_files_dir = os.path.join(params_dir, 'tmp_save_files')
    if not os.path.exists(tmp_files_dir):
        os.makedirs(tmp_files_dir)

    return params_dir, tmp_files_dir


def get_dataset(dataset_dir):
    """
    Get dataset directory, info, and graph.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :return: dataset directory, info, and graph
    :rtype: str, dict, igraph.Graph
    """
    # Validate dataset directory
    if not os.path.exists(dataset_dir):
        logging.error(f'Dataset directory not found.\n{dataset_dir}')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_dir)

    # Get dataset info
    dataset_info_file = os.path.join(dataset_dir, 'dataset_info.pkl')
    if not os.path.isfile(dataset_info_file):
        logging.error(f'Dataset info file not found in directory.\n{dataset_info_file}')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_info_file)
    try:
        f = open(dataset_info_file, 'rb')
    except IOError as e:
        logging.error(f'Could not open dataset info file.\n{dataset_info_file}')
        raise e
    try:
        dataset_info = pickle.load(f)
        # dataset_info = msgpack_load(dataset_info_file)
    except Exception as e:
        logging.error(f'Could not read dataset info file.\n{dataset_info_file}')
        raise e

    # Get graph
    graph_filepath = os.path.join(dataset_dir, dataset_info['gname'])
    if not os.path.isfile(graph_filepath):
        logging.error(f'Graph file not found in directory.\n{graph_filepath}')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), graph_filepath)
    try:
        g = igraph.Graph.Read_Picklez(graph_filepath)
        g.es['weight'] = 1
    except Exception as e:
        logging.error(f'Could not read graph file.\n{graph_filepath}')
        raise e

    return dataset_dir, dataset_info, g


if __name__ == '__main__':
    if len(sys.argv) == 2:
        try:
            # Learn parameters
            learn_parameters(*get_dataset(dataset_dir=sys.argv[1]))
        except Exception as e:
            logging.error('Learn parameters failed!')
            raise e
    else:
        logging.error('Required parameters: dataset directory.')