# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch


def _get_module_stack(node):
    if "nn_module_stack" not in node.meta:
        # if 'fwd_nn_module_stack' in node.meta:
        #    return list(node.meta['fwd_nn_module_stack'].values())
        return []
    return list(node.meta["nn_module_stack"].values())


def _addindent(s_, num_spaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Container:
    def __init__(self, name, klass):
        self.name = name
        self.klass = klass
        self.data = []
        # self.children = defaultdict(Container)
        self.children = {}

    def append(self, data):
        self.data.append(data)

    def get_child(self, module_stack, klass=None):
        if module_stack not in self.children:
            new_stack = Container(module_stack, klass)
            self.children[module_stack] = new_stack
        return self.children[module_stack]

    def __getitem__(self, name):
        return self.children[name]

    def __getattr__(self, name):
        return self.children[name]

    def __repr__(self):
        child_lines = []
        for name, child in self.children.items():
            mod_str = repr(child)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + name + "): " + mod_str)
        main_str = self.klass + "("
        if child_lines:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str

    def graph_view(self):
        return _make_subgraph(self.data)


def _clean_stack_name(val):
    # TODO: is this still needed?
    name = (
        val.replace("L['self']", "Model")
        .replace("_modules['", "")
        .replace("['", ".")
        .replace("']", "")
    )
    return name


def _find_key_nodes(nodes):
    root = []
    outputs = []
    nodes_set = set(nodes)
    for node in nodes:
        for x in node.all_input_nodes:
            if x not in nodes_set:
                root.append(x)
        if all(x not in nodes_set for x in node.users):
            outputs.append(node)
    return root, outputs


def _make_subgraph(nodes):
    placeholders, outputs = _find_key_nodes(nodes)

    new_graph = torch.fx.Graph()
    env = {}

    # pyre-ignore
    def env_lookup(x: torch.fx.Node) -> torch.fx.Node:
        assert x in env, f"Dependent node {x} not in env when creating downstream node"
        return env[x]

    # pyre-ignore
    def node_copy(node, arg_transform) -> torch.fx.Node:
        if node not in env:
            new_node = new_graph.node_copy(node, arg_transform=arg_transform)
            env[node] = new_node
        else:
            new_node = env[node]
        return new_node

    for node in placeholders:
        env[node] = new_graph.placeholder(node.name)

    for node in nodes:
        if node in placeholders:
            continue
        else:
            new_node = node_copy(node, env_lookup)
            new_node.meta = node.meta.copy()

    out_node = [env[x] for x in outputs]
    new_graph.output(out_node)
    return new_graph


def make_graph_view(graph):
    """
    Make a graph view from the fx.Graph. This is a tree structure that
    represents the module hierarchy of the graph, and enables us to
    easily find the nodes that belong to each module, and gives a slightly
    easier way of visualize different parts of the graph by extracting
    subgraphs that belong to a particular module FQN.

    For example, if we have the following model with module hierarchy:

    Transformer(
        (tok_embeddings): Embedding(128256, 4096)
        (layers): ModuleDict(
            (0): TransformerBlock(
            (attention): Attention(
                (wq): Linear(in_features=4096, out_features=4096, bias=False)
                (wk): Linear(in_features=4096, out_features=1024, bias=False)
                (wv): Linear(in_features=4096, out_features=1024, bias=False)
                (wo): Linear(in_features=4096, out_features=4096, bias=False)
                (sdpa): ScaledDotProductAttention()
            )
            (feed_forward): FeedForward(
                (w1): Linear(in_features=4096, out_features=14336, bias=False)
                (w2): Linear(in_features=14336, out_features=4096, bias=False)
                (w3): Linear(in_features=4096, out_features=14336, bias=False)
            )
            (attention_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            (ffn_norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
            )
        )
        (norm): RMSNorm((4096,), eps=1e-05, elementwise_affine=True)
        (output): Linear(in_features=4096, out_features=128256, bias=False)
    )

    Then we can get a GraphView for the fx.Graph that enables us to do

    >>> graph_view = make_graph_view(graph)
    >>> subgraph = graph_view.layers['0'].attention.graph_view()

    where subgraph is a fx.Graph that contains all the nodes that belong to
    Transformer.layers['0'].attention, and whose inputs are all inputs to this
    region of the graph, and whose outputs are all outputs of this region of
    the graph. This returns a new graph with new nodes, so we shouldn't use it
    for graph manipulations, but it is useful to visualize what a particular
    part of a larger graph looks like.

    Additionally, you can also query the original nodes in that region with
    `graph_view.layers['0'].attention.data`, which returns a list of all the
    nodes that belong to Transformer.layers['0'].attention.
    """
    nodes = list(graph.nodes)
    nodes_by_module_stack_root = None
    for node in nodes:
        # TODO: handle cases where there is no module stack (i.e., loop is empty and node is not added)
        for module_stack, module_class in _get_module_stack(node):
            module_stack = _clean_stack_name(module_stack)
            nodes_by_module_stack = nodes_by_module_stack_root
            for name in module_stack.split("."):
                if nodes_by_module_stack is None:
                    nodes_by_module_stack = Container(name, module_class)
                    nodes_by_module_stack_root = nodes_by_module_stack
                new_stack = nodes_by_module_stack.get_child(name, module_class)
                nodes_by_module_stack.data.append(node)
                nodes_by_module_stack = new_stack
    return nodes_by_module_stack_root
