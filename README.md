# DYMOND

**Paper Link**: [DYMOND: DYnamic MOtif-NoDes Network Generative Model](https://gisellezeno.com/files/WWW_2021-DYMOND-GZeno.pdf)  
**Authors**: Giselle Zeno, Timothy La Font, Jennifer Neville

## Abstract 

Motifs, which have been established as building blocks for network structure, move beyond pair-wise connections to capture longer-range correlations in connections and activity. In spite of this, there are few generative graph models that consider higher-order network structures and even fewer that focus on using motifs in models of dynamic graphs. Most existing generative models for temporal graphs strictly grow the networks via edge addition, and the models are evaluated using static graph structure metrics---which do not adequately capture the temporal behavior of the network. To address these issues, in this work we propose DYnamic MOtif-NoDes (DYMOND)---a generative model that considers (i) the dynamic changes in overall graph structure using temporal motif activity and (ii) the roles nodes play in motifs (e.g., one node plays the hub role in a wedge, while the remaining two act as spokes). We compare DYMOND to three dynamic graph generative model baselines on real-world networks and show that DYMOND performs better at generating graph structure and node behavior similar to the observed network. We also propose a new methodology to adapt graph structure metrics to better evaluate the temporal aspect of the network. These metrics take into account the changes in overall graph structure and the individual nodes' behavior over time.

## Citation

    Giselle Zeno, Timothy La Fond, and Jennifer Neville. 2021. DYMOND: DYnamic MOtif-NoDes Network Generative Model. In Proceedings of the Web Conference 2021 (WWW ‘21), April 19–23, 2021, Ljubljana, Slovenia.