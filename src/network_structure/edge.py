# src/network_structure/edge.py

class Edge:
    def __init__(self, parent: 'Node', child: 'Node'):
        self.parent = parent
        self.child = child

    def __str__(self) -> str:
        return f"Edge({self.parent.id} -> {self.child.id})"

    def __repr__(self) -> str:
        return self.__str__()