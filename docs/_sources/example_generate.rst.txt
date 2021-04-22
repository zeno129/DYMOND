Generating Networks
===================

Code
----

After learning the parameters, you can generate a dynamic network by running ``generate_dynamic_graph.py`` directly on the command line

.. code-block:: bash

    python generate_dynamic_graph.py 'path/to/dataset_directory' 'dataset name' 'num. timesteps'

or from your own Python code

.. code-block:: python

    from generate_dynamic_graph import dymond_generate

    dymond_generate(dataset_dir, dataset_name, num_timesteps)


Output files
------------

Generated graph will be saved to ``path/to/dataset_directory/learned_parameters/generated_graph/generated_graph.pklz``.