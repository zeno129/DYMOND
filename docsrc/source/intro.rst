DYMOND Paper
------------
- **PDF:** `DYMOND: DYnamic MOtif-NoDes Network Generative Model <https://gisellezeno.com/files/WWW_2021-DYMOND-GZeno.pdf>`_
- **Video:** `Presentation at The Web Conference 2021 <https://youtu.be/sp2sv4zl3CI>`_
- **Authors:** Giselle Zeno, Timothy La Fond, Jennifer Neville

Citation
--------

.. code-block:: bibtex

    @inproceedings{Zeno2021,
      title = {{{DYMOND}}: {{DYnamic MOtif}}-{{NoDes Network Generative Model}}},
      booktitle = {Proceedings of the {{Web Conference}} 2021 ({{WWW}} '21)},
      author = {Zeno, Giselle and {La Fond}, Timothy and Neville, Jennifer},
      year = {2021},
      pages = {12},
      publisher = {{ACM, New York, NY, USA}},
      address = {{Ljubljana, Slovenia}},
      doi = {10.1145/3442381.3450102}
    }

Abstract
--------
Motifs, which have been established as building blocks for network structure, move beyond pair-wise connections to capture longer-range correlations in connections and activity. In spite of this, there are few generative graph models that consider higher-order network structures and even fewer that focus on using motifs in models of dynamic graphs. Most existing generative models for temporal graphs strictly grow the networks via edge addition, and the models are evaluated using static graph structure metrics—which do not adequately capture the temporal behavior of the network. To address these issues, in this work we propose DYnamic MOtif-NoDes (DYMOND)—a generative model that considers (i) the dynamic changes in overall graph structure using temporal motif activity and (ii) the roles nodes play in motifs (e.g., one node plays the hub role in a wedge, while the remaining two act as spokes). We compare DYMOND to three dynamic graph generative model baselines on real-world networks and show that DYMOND performs better at generating graph structure and node behavior similar to the observed network. We also propose a new methodology to adapt graph structure metrics to better evaluate the temporal aspect of the network. These metrics take into account the changes in overall graph structure and the individual nodes' behavior over time.