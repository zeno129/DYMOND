import os
import sys
import logging
import errno
import gc
import igraph
import numpy as np
from itertools import combinations
from itertools import product
from itertools import chain
from scipy.special import comb
from ..helpers.gzutils import *


def get_active_nodes(num_timesteps, length_timestep, num_nodes, node_rate, gen_dir):
    """
    Get active notes.

    :param num_timesteps: number of timesteps
    :type num_timesteps: int
    :param length_timestep: length of timestep
    :type length_timestep: int
    :param num_nodes: number of nodes
    :type num_nodes: int
    :param node_rate: node arrival rate
    :type node_rate: dict
    :param gen_dir: directory for graph generation
    :type gen_dir: str
    :return: active nodes per timestep
    :rtype: list
    """
    logging.info('Get active nodes')
    active_nodes_file = os.path.join(gen_dir, f'active_nodes_gen.msg')
    if not file_exists(active_nodes_file):
        last_timestep = 0
        V_A = {t: [] for t in range(num_timesteps)}  # initialize V_A
        X = [np.random.exponential(node_rate['scale'] + node_rate['loc'])
             for n in range(num_nodes)]
        for v in range(num_nodes):
            t = int(np.floor(X[v] / length_timestep))
            if 0 <= t and t <= num_timesteps:
                if t > last_timestep:
                    last_timestep = t
                for j in range(t, num_timesteps):
                    V_A[j].append(v)  # add to nodes that are active at time t

        active_nodes_data = {'V_A': V_A, 'last_timestep': last_timestep}
        msgpack_save(active_nodes_file, active_nodes_data)

    else:
        print('  >> read file')
        active_nodes_data = msgpack_load(active_nodes_file)
        V_A = active_nodes_data['V_A']

    return V_A


def helper_get_active_triplets(new_nodes, old_nodes, t, gen_dir, motifs_t):
    logging.info(f'`helper_get_active_triplets()`', f't={t + 1}')
    triplets, num_triplets = get_active_triplets(new_nodes, old_nodes, t)
    return (m for m in triplets if frozenset(m) not in motifs_t), num_triplets - len(motifs_t)

def get_active_triplets(new_nodes, old_nodes, t):
    logging.info(f'Calculate new active triplets t={t + 1}')
    num_old = len(old_nodes)

    # (1) 3 new nodes
    # -------------------------------------------------------*
    triplets = combinations(new_nodes, 3)
    num_triplets = int(comb(len(new_nodes), 3, repetition=False))

    if num_old > 0:
        # (2) 2 new + 1 old
        # -------------------------------------------------------*
        new2_old1 = ((u, v, w)
                     for ((u, v), w) in product(combinations(new_nodes, 2), old_nodes)
                     if w not in [u, v])

        triplets = chain(triplets, new2_old1)
        num_triplets += len(new_nodes) * ((len(new_nodes) - 1)/2) * len(old_nodes)

        # (3) 1 new + 2 old
        # -------------------------------------------------------*
        new1_old2 = ((u, v, w)
                     for ((u, v), w) in product(combinations(old_nodes, 2), new_nodes)
                     if w not in [u, v])

        triplets = chain(triplets, new1_old2)
        num_triplets += len(old_nodes) * ((len(old_nodes) - 1)/2) * len(new_nodes)

    return triplets, int(num_triplets)


def estimate_motif_type_probs(triplets, role_distr, t):
    """
    Estimate motif type probabilities for active triplets at time t.

    :param triplets: new active triplets at time t
    :type triplets: list
    :param role_distr: node role probabilities
    :type role_distr: dict
    :param t: timestep
    :type t: int
    :return: motif type probabilities
    :rtype: dict
    """
    logging.info('Estimate motif type probabilities')
    motif_type_probs = (helper_estimate_motif_type_probs(role_distr, motif, i) for i in [3,2,1]
                        for motif in triplets)
    return motif_type_probs


def helper_estimate_motif_type_probs(role_distr, motif, motif_type):
    u, v, w = motif

    if motif_type == 3:
        prob_triangle = (role_distr[u]['equal3'] *
                         role_distr[v]['equal3'] *
                         role_distr[w]['equal3'])
        return prob_triangle
    elif motif_type == 2:
        prob_wedge = ((role_distr[u]['hub'] * role_distr[v]['spoke'] * role_distr[w]['spoke']) +
                      (role_distr[u]['spoke'] * role_distr[v]['hub'] * role_distr[w]['spoke']) +
                      (role_distr[u]['spoke'] * role_distr[v]['spoke'] * role_distr[w]['hub']))
        return prob_wedge
    elif motif_type == 1:
        prob_edge = ((role_distr[u]['outlier'] * role_distr[v]['equal2'] * role_distr[w]['equal2']) +
                     (role_distr[u]['equal2'] * role_distr[v]['outlier'] * role_distr[w]['equal2']) +
                     (role_distr[u]['equal2'] * role_distr[v]['equal2'] * role_distr[w]['outlier']))
        return prob_edge


def sample_motifs(new_nodes, old_nodes,
                  motif_props,
                  role_distr, role_counts,
                  node_roles_assigned,
                  type_counts,
                  motif_type_rates,
                  t, num_timesteps, gen_dir):
    """
    Sample motifs from new active triplets at time t.

    :param new_nodes: new active nodes
    :type new_nodes: list
    :param old_nodes: previous active nodes
    :type old_nodes: list
    :param motif_props: motif type proportions
    :type motif_props: dict
    :param role_distr: node role probabilities
    :type role_distr: dict
    :param role_counts: node role counts
    :type role_counts: dict
    :param node_roles_assigned: node roles assigned
    :type node_roles_assigned: dict
    :param type_counts: node motif type counts
    :type type_counts: dict
    :param motif_type_rates: motif type inter-arrival rates
    :type motif_type_rates: dict
    :param t: timestep
    :type t: int
    :param num_timesteps: number of timesteps
    :type num_timesteps: int
    :param gen_dir: directory for generated graph
    :type gen_dir: str
    :return: data for motifs sampled at time t
    :rtype: dict
    """
    gen_dir_t = os.path.join(gen_dir, f't_{t+1}')
    motifs_t_file = os.path.join(gen_dir_t, f'motif_data--t_{t+1}.msg')

    logging.info(f'Sample motifs t={t + 1}')
    if not file_exists(motifs_t_file):
        m_types = [3, 2, 1]

        motifs_t = set([])
        motifs_t_show = set([])
        motif_types_t = {}
        motif_timesteps_t = {}
        motif_rates_t = {}
        roles_motifs = {}

        triplets, num_triplets = get_active_triplets(new_nodes, old_nodes, t)
        remaining = triplets

        # Calc expected counts for motif types
        exp_counts = {i: int(np.round(motif_props[i] * num_triplets))
                      for i in m_types}

        for i in m_types:
            if exp_counts[i] > 0:
                probabilities = [helper_estimate_motif_type_probs(role_distr, motif, motif_type=i) *
                                 np.mean([type_counts[node][i]
                                          for node in motif])
                                 for motif in remaining]
                total = sum(probabilities)

                if total > 0.0:
                    sample_size = exp_counts[i]
                    probabilities = np.divide(probabilities, total)
                    num_nonzero = np.count_nonzero(probabilities)
                    remaining, num_remaining = helper_get_active_triplets(new_nodes, old_nodes, t, gen_dir, motifs_t)

                    if sample_size > num_nonzero:
                        logging.warning(f'Could not sample expected counts for motif type i={i} and t={t + 1}')
                        motifs_i = set([frozenset(m)
                                        for idx, m in enumerate(remaining)
                                        if probabilities[idx] > 0])
                        del probabilities
                        gc.collect()
                    else:
                        # Sample motifs
                        motifs_i_idx = list(np.random.choice(range(num_remaining),
                                                             sample_size,
                                                             p=probabilities,
                                                             replace=False))
                        del probabilities
                        gc.collect()

                        motifs_i = set([frozenset(m)
                                        for idx_m, m in enumerate(remaining)
                                        if idx_m in motifs_i_idx])

                    motifs_t.update(motifs_i)
                    remaining, num_remaining = helper_get_active_triplets(new_nodes, old_nodes, t, gen_dir, motifs_t)
                    # Save motif types
                    motif_types_t_i = {motif: i for motif in motifs_i}
                    motif_types_t = {**motif_types_t, **motif_types_t_i}

                    # Sample motif rates and timesteps
                    motif_timesteps_i, motif_rates_i = sample_motif_timesteps(num_timesteps,
                                                                              motifs_i,
                                                                              i,
                                                                              motif_type_rates,
                                                                              t,
                                                                              gen_dir)
                    motif_timesteps_t = {**motif_timesteps_t, **motif_timesteps_i}
                    motif_rates_t = {**motif_rates_t, **motif_rates_i}
                    motifs_i_show = set([m for m in motifs_i if len(motif_timesteps_i[m]) > 0])
                    motifs_t_show = motifs_t_show | motifs_i_show

                    role_distr, role_counts, roles_motifs_i = sample_node_roles(motifs_i_show, i, motif_timesteps_i,
                                                                                role_distr, role_counts,
                                                                                gen_dir, gen_dir_t, t)
                    roles_motifs = {**roles_motifs, **roles_motifs_i}

        motifs_t_data = {'motifs_t': motifs_t,
                         'motifs_t_show': motifs_t_show,
                         'motif_types_t': motif_types_t,
                         'motif_timesteps_t': motif_timesteps_t,
                         'motif_rates_t': motif_rates_t,
                         'roles_motifs': roles_motifs,
                         'role_counts': role_counts,
                         'role_distr': role_distr,
                         'node_roles_assigned': node_roles_assigned
                         }
        msgpack_save(motifs_t_file, motifs_t_data)
    else:
        logging.info('Read save file')
        motifs_t_data = msgpack_load(motifs_t_file)
    return motifs_t_data


def sample_node_roles(motifs, motif_type, motif_timesteps,
                      role_distr, role_counts,
                      gen_dir, gen_dir_t, t):
    """
    Sample node roles for motifs

    :param motifs: motifs
    :type motifs: list
    :param motif_type: motif types
    :type motif_type: dict
    :param motif_timesteps: timesteps motifs appear in
    :type motif_timesteps: dict
    :param role_distr: node role probabilities
    :type role_distr: dict
    :param role_counts: node role counts
    :type role_counts: dict
    :param gen_dir: directory for generated graph
    :type gen_dir: str
    :param gen_dir_t: directory for graph snapshot t
    :type gen_dir_t: str
    :param t: timestep
    :type t: int
    :return: node role probabilities, counts and roles in motifs
    :rtype: dict, dict, dict
    """
    roles_motifs = {motif: {node: None for node in motif}
                    for motif in motifs}
    nodes_to_update = set()

    for motif in motifs:
        u, v, w = motif

        if motif_type == 3:  # triangle
            for node in motif:
                roles_motifs[motif][node] = 'equal3'
                role_counts = update_role_counts(role_counts, node, 'equal3', motif_timesteps[motif])
        elif motif_type == 2:  # wedge
            hub_probs = [float(role_distr[n]['hub']) for n in [u, v, w]]
            total = sum(hub_probs)
            tmp = [p < 0 for p in hub_probs]
            if any(tmp):
                print(f'hub_probs = {hub_probs}')
            assert not any(tmp)
            # sample hub
            if total > 0:
                probabilities = [p/total for p in hub_probs]
                hub = int(np.random.choice([u, v, w], p=probabilities))
            else:
                hub = int(np.random.choice([u, v, w]))
            # others are spokes
            spokes = [n for n in [u, v, w] if n != hub]
            # update counts for distr
            roles_motifs[motif][hub] = 'hub'
            role_counts = update_role_counts(role_counts, hub, 'hub', motif_timesteps[motif])
            for spoke in spokes:
                roles_motifs[motif][spoke] ='spoke'
                role_counts = update_role_counts(role_counts, spoke, 'spoke', motif_timesteps[motif])
        elif motif_type == 1:  # 1-edge
            outlier_probs = [float(role_distr[n]['outlier']) for n in [u, v, w]]
            total = sum(outlier_probs)
            if any([p < 0 for p in outlier_probs]):
                print(f'outlier_probs = {outlier_probs}')
            # sample outlier
            if total > 0:  # TODO: assert total > 0
                probabilities = [p / total for p in outlier_probs]
                outlier = int(np.random.choice([u, v, w], p=probabilities))
            else:
                outlier = int(np.random.choice([u, v, w]))
            # others are equal2
            equal2 = [n for n in [u, v, w] if n != outlier]
            # update counts for distr
            roles_motifs[motif][outlier] = 'outlier'
            role_counts = update_role_counts(role_counts, outlier, 'outlier', motif_timesteps[motif])
            for eq2 in equal2:
                roles_motifs[motif][eq2] = 'equal2'
                role_counts = update_role_counts(role_counts, eq2, 'equal2', motif_timesteps[motif])
        nodes_to_update = nodes_to_update | set(motif)
    # Update role distribution
    role_distr = update_role_distr(list(nodes_to_update), role_distr, role_counts, gen_dir, t, gen_dir_t)

    return role_distr, role_counts, roles_motifs


def update_role_counts(role_counts, u, role_u, motif_timesteps):
    """
    Update node role counts

    :param role_counts: node role counts
    :type role_counts: dict
    :param u: node to update counts for
    :type u: int
    :param role_u: node u's role
    :type role_u: str
    :param motif_timesteps: timesteps motif appears in
    :type motif_timesteps: list
    :return: updated role counts
    :rtype: dict
    """
    role_counts[u][role_u] -= len(motif_timesteps)
    return role_counts


def update_role_distr(nodes, role_distr, role_counts, gen_dir, t, gen_dir_t=None):
    """
    Update node role probabilities

    :param nodes: nodes
    :type nodes: list
    :param role_distr: node role probabilities
    :type role_distr: dict
    :param role_counts: node role counts
    :type role_counts: dict
    :param gen_dir: directory for generated graph
    :type gen_dir: str
    :param t: timestep
    :type t: int
    :param gen_dir_t: directory for generated graph snapshot t
    :type gen_dir_t: str
    :return: updated node role probabilities
    :rtype: dict
    """
    logging.info(f'Update node role probabilities t={t + 1}')
    roles = ['equal3', 'hub', 'spoke', 'equal2', 'outlier']

    for node in nodes:
        counts = {r: role_counts[node][r] if role_counts[node][r] >= 0 else 0 for r in roles}
        total = sum([counts[r] for r in roles])
        role_distr[node] = {r: counts[r]/total if total > 0 and counts[r] > 0 else 0
                            for r in roles}

    role_distr_file = os.path.join(gen_dir, f'role_distr.msg')
    msgpack_save(role_distr_file, {'role_distr': role_distr,
                                  'role_counts': role_counts})
    if gen_dir_t:
        # save a backup for last timestep generated
        role_distr_file = os.path.join(gen_dir_t, f'role_distr.msg')
        msgpack_save(role_distr_file, {'role_distr': role_distr,
                                      'role_counts': role_counts})
    return role_distr


def get_role_distr(gen_dir, model_params):
    """
    Get node role probabilities

    :param gen_dir: directory for generated graph
    :type gen_dir: str
    :param model_params: model parameters
    :type model_params: dict
    :return: node role probabilities
    :rtype: dict
    """
    role_distr_file = os.path.join(gen_dir, f'role_distr.msg')
    if file_exists(role_distr_file):  # load updated distribution and counts
        roles_data = msgpack_load(role_distr_file)
        role_distr = roles_data['role_distr']
        role_counts = roles_data['role_counts']
    else:  # load input params. distribution and counts
        role_distr = model_params['node roles distr']
        role_counts = model_params['node roles counts']
        role_distr = update_role_distr(list(role_counts.keys()),
                                       role_distr, role_counts, gen_dir, -2)

    return role_distr, role_counts


def get_motif_edges(motifs, motif_types, roles_motifs, roles_assigned, t, gen_dir):
    """
    Get motif edges

    :param motifs: motifs
    :type motifs: list
    :param motif_types: motif types
    :type motif_types: dict
    :param roles_motifs: node roles
    :type roles_motifs: dict
    :param roles_assigned: node roles assigned counts
    :type roles_assigned: dict
    :param t: timestep
    :type t: int
    :param gen_dir: directory for generated graph
    :type gen_dir: str
    :return: motif edges, node roles assigned
    :rtype: dict, dict
    """
    logging.info('Get motif edges')
    gen_dir_t = os.path.join(gen_dir, f't_{t+1}')
    motif_edges_file = os.path.join(gen_dir_t, f'motif_edges_t_{t+1}.msg')
    roles_assigned_file = os.path.join(gen_dir, f'roles_assigned.msg')

    if not file_exists(motif_edges_file):
        motif_edges = {}
        print('  >> sample')
        for motif in motifs:
            u, v, w = motif
            motif_type = motif_types[motif]

            if motif_type == 3:  # triangle
                motif_edges[motif] = list(combinations([u, v, w], 2))
                # all nodes are equal3
                for node in motif:
                    roles_assigned[node]['equal3'] += 1
            elif motif_type == 2:  # wedge
                # get hub
                hub = [node for node in motif if roles_motifs[motif][node] == 'hub']
                assert len(hub) > 0
                hub = hub[0]
                roles_assigned[hub]['hub'] += 1
                # others are spokes
                spokes = [n for n in [u, v, w] if n != hub]
                for spoke in spokes:
                    roles_assigned[spoke]['spoke'] += 1
                # save edges
                motif_edges[motif] = list(product([hub], spokes))
            elif motif_type == 1:  # 1-edge
                # get outlier
                outlier = [node for node in motif if roles_motifs[motif][node] == 'outlier']
                assert len(outlier) > 0
                outlier = outlier[0]
                roles_assigned[outlier]['outlier'] += 1
                # others are equal2
                equal2 = [n for n in [u, v, w] if n != outlier]
                for eq2 in equal2:
                    roles_assigned[eq2]['equal2'] += 1
                # save edges
                motif_edges[motif] = [tuple(equal2)]
        msgpack_save(motif_edges_file, motif_edges)
        msgpack_save(roles_assigned_file, roles_assigned)
    else:
        print('  >> read file')
        motif_edges = msgpack_load(motif_edges_file)
        roles_assigned = msgpack_load(roles_assigned_file)

    return motif_edges, roles_assigned


def sample_motif_timesteps(num_timesteps, motifs, motif_type, motif_type_rates, t, gen_dir):
    """
    Sample timesteps that the motifs will appear in.

    :param num_timesteps: number of timesteps to generate
    :type num_timesteps: int
    :param motifs: motifs
    :type motifs: dict
    :param motif_type: motif types
    :type motif_type: dict
    :param motif_type_rates: motif type inter-arrival rates
    :type motif_type_rates: dict
    :param t: timestep
    :type t: int
    :param gen_dir: directory for generated graph
    :type gen_dir: str
    :return: motif timesteps and inter-arrival rates
    :rtype: dict, dict
    """
    logging.info(f'Sample timesteps new motifs t={t + 1}')
    gen_dir_t = os.path.join(gen_dir, f't_{t+1}')
    motif_timesteps_file = os.path.join(gen_dir_t, f'motif_timesteps_t_{t+1}.msg')

    # Initialization
    motif_timesteps = {m: set([]) for m in motifs}
    motif_rates = {m: None for m in motifs}
    prev_motifs = set([])

    if file_exists(motif_timesteps_file):
        logging.info('Read from save file')
        timestep_data = msgpack_load(motif_timesteps_file)
        motif_timesteps_prev = timestep_data['motif_timesteps']
        motif_timesteps.update(motif_timesteps_prev)
        motif_rates = timestep_data['motif_rates']
        prev_motifs = set(timestep_data['motif_timesteps'].keys())
    motifs_to_do = motifs - prev_motifs

    if motifs_to_do:
        logging.info('Sample timesteps for remaining motifs')
        for motif in motifs_to_do:
            i = motif_type

            # Sample inter-arrival rate
            motif_rates[motif] = np.random.exponential(motif_type_rates[i]['scale'] +
                                                       motif_type_rates[i]['loc'])
            interarrival_scale = 1 / motif_rates[motif]

            curr_time = t  # first timestep motif could appear
            next_time = np.random.exponential(interarrival_scale)
            timestep = int(np.ceil(curr_time + next_time))

            tries = 0  # debugging
            while timestep < num_timesteps:
                motif_timesteps[motif].add(timestep)  # save timestep
                curr_time = timestep
                next_time = np.random.exponential(interarrival_scale)
                timestep = int(np.ceil(curr_time + next_time))

                tries += 1  # debugging
                assert tries < num_timesteps
        msgpack_save(motif_timesteps_file, {'motif_timesteps': motif_timesteps,
                                            'motif_rates': motif_rates})

    return motif_timesteps, motif_rates


def generate_dynamic_graph(num_timesteps,
                           length_timestep,
                           num_nodes,
                           node_rate,
                           motif_props,
                           role_distr,
                           role_counts,
                           type_counts,
                           motif_type_rates,
                           gen_data_dir):
    """
    Generate dynamic graph.

    :param num_timesteps: number of timesteps to generate
    :type num_timesteps: int
    :param length_timestep: length of time
    :type length_timestep: int
    :param num_nodes: number of nodes
    :type num_nodes: int
    :param node_rate: node arrival rate
    :type node_rate: dict
    :param motif_props: motif type proportions
    :type motif_props: dict
    :param role_distr: node role probabilities
    :type role_distr: dict
    :param role_counts: node role counts
    :type role_counts: dict
    :param type_counts: motif type counts
    :type type_counts: dict
    :param motif_type_rates: motif type inter-arrival rates
    :type motif_type_rates: dict
    :param gen_data_dir: directory for generated graph
    :type gen_data_dir: str
    :return: generated graph data
    :rtype: dict
    """
    logging.info('Generate dynamic graph')
    time_code = log_time()

    # Get active nodes
    active_nodes = get_active_nodes(num_timesteps, length_timestep, num_nodes, node_rate, gen_data_dir)

    # Initialization
    motifs = set([])
    motifs_show = set([])
    motif_types = {}
    motif_edges = {}
    motif_timesteps = {}
    motif_rates = {}

    roles = ['equal3', 'hub', 'spoke', 'equal2', 'outlier']
    motif_roles_assigned = {v: {r: 0
                                   for r in roles}
                               for v in range(num_nodes)}
    node_roles_assigned = {v: {t: set([])
                               for t in range(num_timesteps)}
                           for v in range(num_nodes)}

    prev_nodes = []
    for t in range(num_timesteps):
        new_nodes = list(set(active_nodes[t]) - set(prev_nodes))
        if len(new_nodes) > 0:
            logging.info(f'Generate timestep t={t+1:,}')
            gen_dir_t = os.path.join(gen_data_dir, f't_{t+1}')
            if not os.path.exists(gen_dir_t):
                os.makedirs(gen_dir_t)

            # Sample motifs
            motifs_t_data = sample_motifs(new_nodes, prev_nodes, motif_props,
                                          role_distr, role_counts,
                                          node_roles_assigned,
                                          type_counts,
                                          motif_type_rates, t, num_timesteps,
                                          gen_data_dir)

            # Parse data
            motifs_t = motifs_t_data['motifs_t']
            motifs_t_show = motifs_t_data['motifs_t_show']
            motif_types_t = motifs_t_data['motif_types_t']
            motif_timesteps_t = motifs_t_data['motif_timesteps_t']
            motif_rates_t = motifs_t_data['motif_rates_t']
            roles_t = motifs_t_data['roles_motifs']
            role_counts = motifs_t_data['role_counts']
            role_distr = motifs_t_data['role_distr']
            node_roles_assigned = motifs_t_data['node_roles_assigned']

            # Get motif edges
            motif_edges_t, motif_roles_assigned = get_motif_edges(motifs_t_show, motif_types_t,
                                                                  roles_t, motif_roles_assigned,
                                                                  t, gen_data_dir)

            # Update
            motifs = motifs | motifs_t
            motifs_show = motifs_show | motifs_t_show
            motif_types = {**motif_types, **motif_types_t}
            motif_timesteps = {**motif_timesteps, **motif_timesteps_t}
            motif_rates = {**motif_rates, **motif_rates_t}
            motif_edges = {**motif_edges, **motif_edges_t}

            # Next round
            prev_nodes = active_nodes[t]

            total_time = time_code('')
            logging.info(f'Finished generating graph! {total_time}')

    # Create dictionary with the generated graph data
    gen_data = {'num nodes': num_nodes,
                'active nodes': active_nodes,
                'node roles assigned': node_roles_assigned,
                'motif roles assigned ': motif_roles_assigned,
                'motifs': motifs,
                'motifs show': motifs_show,
                'motif types': motif_types,
                'motif edges': motif_edges,
                'num timesteps': num_timesteps,
                'motif timesteps': motif_timesteps,
                'motif rates': motif_rates}

    return gen_data


def get_generated_graph_data(gen_data_dir, num_timesteps, model_params):
    """
    Generate graph data.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :param num_timesteps: number of timesteps to generate
    :type num_timesteps: int
    :param model_params: model parameters
    :type model_params: dict
    :return: generated graph data
    :rtype: dict
    """
    gen_data_file = os.path.join(gen_data_dir, 'gen_data.msg')

    if not file_exists(gen_data_file):
        length_timestep = model_params['L']
        num_nodes = model_params['N']
        node_rate = model_params['node arrivals']
        motif_props = model_params['motif proportions']

        # Get node role probabilities
        role_distr, role_counts = get_role_distr(gen_data_dir, model_params)
        type_counts = model_params['node motif types counts']
        motif_type_rates = model_params['motif interarrivals']

        # Generate dynamic graph
        gen_data = generate_dynamic_graph(num_timesteps,
                                          length_timestep,
                                          num_nodes,
                                          node_rate,
                                          motif_props,
                                          role_distr,
                                          role_counts,
                                          type_counts,
                                          motif_type_rates,
                                          gen_data_dir)

        msgpack_save(gen_data_file, gen_data)
    else:
        logging.info('Read generated data')
        gen_data = msgpack_load(gen_data_file)

    # Check edge data
    if 'graph edges' not in gen_data.keys():
        motifs_show = gen_data['motifs show']
        motif_types = gen_data['motif types']
        motif_edges = gen_data['motif edges']
        motif_timesteps = gen_data['motif timesteps']
        num_timesteps = gen_data['num timesteps']

        # Get graph edges
        graph_edges = {t: set([])
                       for t in range(num_timesteps)}
        counts_per_timestep = {t: {i: 0
                                   for i in range(1, 4)}
                               for t in range(num_timesteps)}

        for motif in motifs_show:
            if motif in motif_timesteps.keys():
                i = motif_types[motif]
                for t in motif_timesteps[motif]:
                    counts_per_timestep[t][i] += 1
                    the_edges = set([tuple(e) for e in motif_edges[motif]])
                    graph_edges[t] = graph_edges[t] | the_edges
        gen_data['graph edges'] = graph_edges

    return gen_data


def create_igraph(gen_data, gen_data_dir):
    """
    Create igraph.Graph object

    :param gen_data: generated graph data
    :type gen_data: dict
    :param gen_data_dir: directory for generated graph
    :type gen_data_dir: str
    """
    logging.info('Create igraph.Graph')
    edges = []
    edge_timesteps = []
    for t in range(gen_data['num timesteps']):
        edges.extend(gen_data['graph edges'][t])
        edge_timesteps.extend([t] * len(gen_data['graph edges'][t]))

    g = igraph.Graph(n=gen_data['num nodes'],
                     directed=False,
                     edges=edges,
                     edge_attrs={'timestep': edge_timesteps}
                     )

    g_path = os.path.join(gen_data_dir, f'generated_graph.pklz')
    g.write_picklez(g_path)
    logging.info(f'Generated graph path:  {os.path.abspath(g_path)}')


def get_directories_generated_graph(dataset_dir):
    """
    Get directories for model parameters and generated graph.

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :return: directories for model parameters and generated graph
    :rtype: str, str
    """
    # Validate parameters directory
    params_dir = os.path.join(dataset_dir, 'learned_parameters')
    if not os.path.exists(params_dir):
        logging.error(f'Dataset parameters directory not found.\n{dataset_dir}')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), params_dir)

    # Directory to save generated graph
    gen_data_dir = os.path.join(params_dir, 'generated_graph')
    if not os.path.exists(gen_data_dir):
        os.makedirs(gen_data_dir)

    return params_dir, gen_data_dir


def get_parameters(params_dir):
    """
    Get model parameters

    :param params_dir: parameters directory
    :type params_dir: str
    :return: model parameters
    :rtype: dict
    """
    logging.info('Read model parameters')
    model_params_file = os.path.join(params_dir, 'model_params.msg')
    model_params = msgpack_load(model_params_file)

    return model_params


def dymond_generate(dataset_dir, num_timesteps):
    """
    Run graph generation

    :param dataset_dir: dataset directory
    :type dataset_dir: str
    :param dataset_name: dataset name
    :type dataset_name: str
    :param num_timesteps: number of timesteps to generate
    :type num_timesteps: int
    """
    # Get directories for model parameters and generated graph
    params_dir, gen_data_dir = get_directories_generated_graph(dataset_dir)

    # Get model parameters
    model_params = get_parameters(params_dir)

    # Generate graph
    gen_data = get_generated_graph_data(gen_data_dir, num_timesteps, model_params)

    # Create igraph
    create_igraph(gen_data, gen_data_dir)


if __name__ == '__main__':
    gc.enable()

    if len(sys.argv) == 3:
        try:
            dymond_generate(dataset_dir=sys.argv[1], num_timesteps=int(sys.argv[2]))
        except Exception as e:
            logging.error('Graph generation failed!')
            raise e
    else:
        logging.error('Required parameters: dataset path and number of timesteps to generate.')
