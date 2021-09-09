#TODO: refactor with a single class and a generic comparing method
class MinHeap:

    def __init__(self, fun=lambda x: x):
        self.__heap = []
        self.__fun = fun

    def __iter__(self):
        return iter(self.__heap)

    def __next__(self):
        return next(self.__heap)

    def min(self):
        return self.__heap[0]

    def __len__(self):
        return len(self.__heap)

    def empty(self):
        return len(self.__heap) == 0

    def extract_min(self):
        x = self.__heap[0]
        self.__heap[0] = self.__heap[-1]
        self.__heap.pop(-1)
        self._heapify(0)
        return x

    def _get_left(self, i):
        return 2 * i + 1

    def _get_right(self, i):
        return 2 * i + 2

    def _get_parent(self, i):
        return (i - 1) // 2

    def get_elements(self):
        return self.__heap

    def _heapify(self, i):
        left, right = self._get_left(i), self._get_right(i)
        if left >= len(self.__heap):
            return
        min_child = left
        if right < len(self.__heap):
            l_val, r_val = self.__heap[left], self.__heap[right]
            if self.__fun(l_val) > self.__fun(r_val):
                min_child = right
        val = self.__heap[i]
        min_val = self.__heap[min_child]
        if self.__fun(val) > self.__fun(min_val):
            self.__heap[i], self.__heap[min_child] = min_val, val
            self._heapify(min_child)

    def insert(self, x):
        self.__heap.append(x)
        node = len(self.__heap) - 1
        parent = self._get_parent(node)
        n_val, p_val = self.__heap[node], self.__heap[parent]
        while parent >= 0 and self.__fun(n_val) < self.__fun(p_val):
            self.__heap[node], self.__heap[parent] = p_val, n_val
            node = parent
            parent = self._get_parent(node)
            p_val = self.__heap[parent]

    def update(self, values):
        for x in values:
            self.insert(x)


class MaxHeap:

    def __init__(self, fun=lambda x: x):
        self.__heap = []
        self.__fun = fun

    def __iter__(self):
        return iter(self.__heap)

    def __next__(self):
        return next(self.__heap)

    def max(self):
        return self.__heap[0]

    def extract_max(self):
        x = self.__heap[0]
        self.__heap[0] = self.__heap[-1]
        self.__heap.pop(-1)
        self._heapify(0)
        return x

    def _get_left(self, i):
        return 2 * i + 1

    def _get_right(self, i):
        return 2 * i + 2

    def _get_parent(self, i):
        return (i - 1) // 2

    def get_elements(self):
        return self.__heap

    def empty(self):
        return len(self.__heap) == 0

    def __len__(self):
        return len(self.__heap)

    def _heapify(self, i):
        left, right = self._get_left(i), self._get_right(i)
        if left >= len(self.__heap):
            return
        max_child = left
        if right < len(self.__heap):
            l_val, r_val = self.__heap[left], self.__heap[right]
            if self.__fun(l_val) < self.__fun(r_val):
                max_child = right
        val = self.__heap[i]
        max_val = self.__heap[max_child]
        if self.__fun(val) < self.__fun(max_val):
            self.__heap[i], self.__heap[max_child] = max_val, val
            self._heapify(max_child)

    def insert(self, x):
        self.__heap.append(x)
        node = len(self.__heap) - 1
        parent = self._get_parent(node)
        n_val, p_val = self.__heap[node], self.__heap[parent]
        while parent >= 0 and self.__fun(n_val) > self.__fun(p_val):
            self.__heap[node], self.__heap[parent] = p_val, n_val
            node = parent
            parent = self._get_parent(node)
            p_val = self.__heap[parent]

    def update(self, values):
        for x in values:
            self.insert(x)

