def khop_subgraph(
    graph, nodes, k, *,fanout=5, relabel_nodes=True, store_ids=True, output_device=None
):
    """Return the subgraph induced by k-hop neighborhood of the specified node(s).

    We can expand a set of nodes by including the successors and predecessor of them. From a
    specified node set, a k-hop subgraph is obtained by first repeating the node set
    expansion for k times and then creating a node induced subgraph. In addition to
    extracting the subgraph, DGL also copies the features of the extracted nodes and
    edges to the resulting graph. The copy is *lazy* and incurs data movement only
    when needed.

    If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
    them as the resulting graph. Thus the resulting graph has the same set of relations
    as the input one.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    nodes : nodes or dict[str, nodes]
        The starting node(s) to expand, which cannot have any duplicate value. The result
        will be undefined otherwise. The allowed formats are:

        * Int: ID of a single node.
        * Int Tensor: Each element is a node ID. The tensor must have the same device
          type and ID data type as the graph's.
        * iterable[int]: Each element is a node ID.

        If the graph is homogeneous, one can directly pass the above formats.
        Otherwise, the argument must be a dictionary with keys being node types
        and values being the node IDs in the above formats.
    k : int
        The number of hops.
    relabel_nodes : bool, optional
        If True, it will remove the isolated nodes and relabel the rest nodes in the
        extracted subgraph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
        resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
        also store the raw IDs of the extracted nodes in the ``ndata`` of the resulting
        graph under name ``dgl.NID``.
    output_device : Framework-specific device context object, optional
        The output device.  Default is the same as the input graph.

    Returns
    -------
    DGLGraph
        The subgraph.
    Tensor or dict[str, Tensor], optional
        The new IDs of the input :attr:`nodes` after node relabeling. This is returned
        only when :attr:`relabel_nodes` is True. It is in the same form as :attr:`nodes`.

    """
    if graph.is_block:
        raise DGLError("Extracting subgraph of a block graph is not allowed.")

    is_mapping = isinstance(nodes, Mapping)
    if not is_mapping:
        assert (
            len(graph.ntypes) == 1
        ), "need a dict of node type and IDs for graph with multiple node types"
        nodes = {graph.ntypes[0]: nodes}

    for nty, nty_nodes in nodes.items():
        nodes[nty] = utils.prepare_tensor(
            graph, nty_nodes, 'nodes["{}"]'.format(nty)
        )

    last_hop_nodes = nodes
    k_hop_nodes_ = [last_hop_nodes]
    device = context_of(nodes)
    place_holder = F.copy_to(F.tensor([], dtype=graph.idtype), device)
    for _ in range(k):
        current_hop_nodes = {nty: [] for nty in graph.ntypes}
        # add outgoing nbrs
        for cetype in graph.canonical_etypes:
            srctype, _, dsttype = cetype
            _, out_nbrs = graph.out_edges(
                last_hop_nodes.get(srctype, place_holder), etype=cetype
            )
            if fanout<len(out_nbrs):
                current_hop_nodes[dsttype].append(random.sample(out_nbrs,fanout))
            else:
                current_hop_nodes[dsttype].append(out_nbrs)
        # add incoming nbrs
        for cetype in graph.canonical_etypes:
            srctype, _, dsttype = cetype
            in_nbrs, _ = graph.in_edges(
                last_hop_nodes.get(dsttype, place_holder), etype=cetype
            )
            if fanout<len(in_nbrs):
                current_hop_nodes[srctype].append(random.sample(in_nbrs,fanout))
            else:
                current_hop_nodes[srctype].append(in_nbrs)
        for nty in graph.ntypes:
            if len(current_hop_nodes[nty]) == 0:
                current_hop_nodes[nty] = place_holder
                continue
            current_hop_nodes[nty] = F.unique(
                F.cat(current_hop_nodes[nty], dim=0)
            )
        k_hop_nodes_.append(current_hop_nodes)
        last_hop_nodes = current_hop_nodes

    k_hop_nodes = dict()
    inverse_indices = dict()
    for nty in graph.ntypes:
        k_hop_nodes[nty], inverse_indices[nty] = F.unique(
            F.cat(
                [
                    hop_nodes.get(nty, place_holder)
                    for hop_nodes in k_hop_nodes_
                ],
                dim=0,
            ),
            return_inverse=True,
        )


    sub_g = node_subgraph(
        graph, k_hop_nodes, relabel_nodes=relabel_nodes, store_ids=store_ids
    )
    if output_device is not None:
        sub_g = sub_g.to(output_device)
    if relabel_nodes:
        if is_mapping:
            seed_inverse_indices = dict()
            for nty in nodes:
                seed_inverse_indices[nty] = F.slice_axis(
                    inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty])
                )
        else:
            seed_inverse_indices = F.slice_axis(
                inverse_indices[nty], axis=0, begin=0, end=len(nodes[nty])
            )
        if output_device is not None:
            seed_inverse_indices = recursive_apply(
                seed_inverse_indices, lambda x: F.copy_to(x, output_device)
            )
        return sub_g, seed_inverse_indices
    else:
        return sub_g


DGLGraph.khop_subgraph = utils.alias_func(khop_subgraph)