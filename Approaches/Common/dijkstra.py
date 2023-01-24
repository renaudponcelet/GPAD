from collections import defaultdict
from heapq import heapify, heappush, heappop


def weight(node2):
    return node2[1]


def dijkstra(graph, origin, destination):
    D = {}
    P = {}
    Q = PriorityDict()
    Q[origin] = 0
    for v in Q:
        D[v] = Q[v]
        if v == destination:
            break
        edges = graph[v]
        for e in edges:
            e_length = D[v] + weight(e)
            if e[0] in D:
                if e_length < D[e[0]]:
                    raise ValueError
            elif e[0] not in Q or e_length < Q[e[0]]:
                Q[e[0]] = e_length
                P[e[0]] = v
    return D, P


class PriorityDict(dict):
    """Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is that priorities
    of items can be efficiently updated (amortized O(1)) using code as
    'thedict[item] = new_priority.'

    Note that this is a modified version of
    https://gist.github.com/matteodellamico/4451520 where sorted_iter() has
    been replaced with the destructive sorted iterator __iter__ from
    https://gist.github.com/anonymous/4435950
    """
    def __init__(self, *args, **kwargs):
        super(PriorityDict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in iter(self.items())]
        heapify(self._heap)

    def smallest(self):
        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        super(PriorityDict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            self._rebuild_heap()

    def setdefault(self, *args, **kwargs):
        if len(args) == 2:
            key = args[0]
            val = args[1]
        else:
            raise KeyError
        if 'key' in kwargs:
            key = kwargs['key']
        if 'val' in kwargs:
            val = kwargs['val']
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        super(PriorityDict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def __iter__(self):
        def iter_fn():
            while len(self) > 0:
                x = self.smallest()
                yield x
                del self[x]
        return iter_fn()


class Graph:
    def __init__(self):
        self.dict = defaultdict(set)

    def add_edge(self, n1, n2, dist):
        self.dict[n1].add((n2, dist))

    def remove_edge(self, n):
        del self.dict[n]

    def shortest_path(self, origin, destination):
        D, P = dijkstra(self.dict, origin, destination)
        path = []
        try:
            while 1:
                path.append(destination)
                if destination == origin:
                    break
                destination = P[destination]
        except KeyError:
            return None
        path.reverse()
        return path
