Learning Parameters
===================

Code
----

To learn the model parameters, you can run ``learn_parameters.py`` directly on the command line

.. code-block:: bash

    python learn_parameters.py 'path/to/dataset_directory'

or from your own Python code

.. code-block:: python

    from learn_parameters import get_dataset, learn_parameters

    dataset_dir, dataset_info, g = get_dataset(dataset_dir='path/to/dataset_directory')
    learn_parameters(dataset_dir, dataset_info, g)


Input files
-----------
You'll need to create the following files inside your dataset directory.

Graph
^^^^^
Create igraph file for dataset as follows:

.. code-block:: python

    import igraph

    # Create igraph
    g = igraph.Graph(n=len(nodes),  # nodes is a a list with the nodes
                     directed=False,
                     edges=edge_list,  # list of edges (not unique), with indices in node list (u,v)
                     edge_attrs={'timestep': edge_timesteps}  # list of timesteps, one for each edge in edge_list
                     )

    # Annotate with time when nodes become active
    for v in g.vs:
        v['nid'] = f'nid-{v.index}'  # Annotate with original index
        neighbors = list(set([u for u in g.neighbors(v)]))
        if len(neighbors) > 0:
            v_edges = g.es.select(_between=([v.index], neighbors))
            v['active'] = min(v_edges['timestep'])

    # Save to file
    graph_filename = 'dataset_name.pklz'
    g.write_pickle(os.path.join(dataset_dir, graph_filename))


Dataset info
^^^^^^^^^^^^
Create dataset info file as follows:

.. code-block:: python

    import pickle

    dataset_info = {'gname': graph_filename,
                    'L': 1,
                    'N': g.vcount(),
                    'T': len(timesteps),
                    'timesteps': timesteps
    }

    dataset_info_file = os.path.join(dataset_dir, 'dataset_info.pkl')
    output = open(dataset_info_file, 'wb')
    pickle.dump(dataset_info, output)

Output files
------------

Parameters will be saved to ``path/to/dataset_directory/learned_parameters/model_params.msg``.

