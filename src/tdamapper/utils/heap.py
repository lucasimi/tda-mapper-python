def _left(i):
    return 2 * i + 1


def _right(i):
    return 2 * i + 2


def _parent(i):
    return max(0, (i - 1) // 2)


class _HeapNode:

    def __init__(self, key, value):
        self.__key = key
        self.__value = value

    def get(self):
        return self.__key, self.__value

    def __lt__(self, other):
        return self.__key < other

    def __le__(self, other):
        return self.__key <= other

    def __gt__(self, other):
        return self.__key > other

    def __ge__(self, other):
        return self.__key >= other


class MaxHeap:

    def __init__(self):
        self.__heap = []
        self.__iter = None

    def __iter__(self):
        self.__iter = iter(self.__heap)
        return self

    def __next__(self):
        node = next(self.__iter)
        return node.get()

    def __len__(self):
        return len(self.__heap)

    def top(self):
        if not self.__heap:
            return (None, None)
        return self.__heap[0].get()

    def pop(self):
        if not self.__heap:
            return
        max_val = self.__heap[0]
        self.__heap[0] = self.__heap[-1]
        self.__heap.pop()
        self._bubble_down()
        return max_val.get()

    def add(self, key, val):
        self.__heap.append(_HeapNode(key, val))
        self._bubble_up()

    def _get_local_max(self, i):
        heap_len = len(self.__heap)
        left = _left(i)
        right = _right(i)
        if left >= heap_len:
            return i
        if right >= heap_len:
            if self.__heap[i] < self.__heap[left]:
                return left
            return i
        max_child = left
        if self.__heap[left] < self.__heap[right]:
            max_child = right
        if self.__heap[i] < self.__heap[max_child]:
            return max_child
        return i

    def _fix_down(self, i):
        local_max = self._get_local_max(i)
        if i < local_max:
            self.__heap[i], self.__heap[local_max] = self.__heap[local_max], self.__heap[i]
            return local_max
        return i

    def _fix_up(self, i):
        parent = _parent(i)
        if self.__heap[parent] < self.__heap[i]:
            self.__heap[i], self.__heap[parent] = self.__heap[parent], self.__heap[i]
            return parent
        return i

    def _bubble_down(self):
        current = 0
        done = False
        while not done:
            local_max = self._fix_down(current)
            done = current == local_max
            current = local_max

    def _bubble_up(self):
        current = len(self.__heap) - 1
        done = False
        while not done:
            local_max = self._fix_up(current)
            done = local_max == current
            current = local_max
