import numpy as np
from typing import Tuple, List, Any, TypeAlias


# array of shape (N,)
Nodes: TypeAlias = np.typing.NDArray[[-1], np.uint32]
Edges: TypeAlias = np.typing.NDArray[[-1, 2], np.uint32]
GeneralArchitectureReturn: TypeAlias = Tuple[Nodes, Edges]


def make_block(n=2) -> GeneralArchitectureReturn:
    return np.arange(n, dtype=np.uint32), np.array(
        [[i, i + 1] for i in range(n - 1)], dtype=np.uint32
    )  # type: ignore


def make_ring(nodes=2) -> GeneralArchitectureReturn:
    nodes, edges = make_block(nodes)
    edges = np.vstack((edges, [nodes[-1], 0]))
    return nodes, edges  # type: ignore


def make_star(n=2) -> GeneralArchitectureReturn:
    nodes = np.array([0] + [1] * n, dtype=np.uint32)
    edges = np.array([[0, i] for i in range(1, n + 1)], dtype=np.uint32)
    return nodes, edges  # type: ignore


def make_linear() -> GeneralArchitectureReturn:
    return np.array(0), np.array([]).reshape(0, 2)  # type: ignore


def make_branched(b=2, n=5, rng=None) -> GeneralArchitectureReturn:
    if rng is None:
        rng = np.random.default_rng()
    n