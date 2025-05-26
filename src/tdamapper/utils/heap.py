def _left(i):
    return 2 * i + 1


def _right(i):
    return 2 * i + 2


def _parent(i):
    return max(0, (i - 1) // 2)


class _HeapNode:

    def __init__(self, key, value):
        self._key = key
        self._value = value

    def get(self):
        return self._key, self._value

    def __lt__(self, other):
        return self._key < other._key

    def __le__(self, other):
        return self._key <= other._key

    def __gt__(self, other):
        return self._key > other._key

    def __ge__(self, other):
        return self._key >= other._key


class MaxHeap:

    def __init__(self):
        self._heap = []
        self._iter = None

    def __iter__(self):
        self._iter = iter(self._heap)
        return self

    def __next__(self):
        node = next(self._iter)
        return node.get()

    def __len__(self):
        return len(self._heap)

    def top(self):
        if not self._heap:
            return (None, None)
        return self._heap[0].get()

    def pop(self):
        if not self._heap:
            return
        max_val = self._heap[0]
        self._heap[0] = self._heap[-1]
        self._heap.pop()
        self._bubble_down()
        return max_val.get()

    def add(self, key, val):
        self._heap.append(_HeapNode(key, val))
        self._bubble_up()

    def _get_local_max(self, i):
        heap_len = len(self._heap)
        left = _left(i)
        right = _right(i)
        if left >= heap_len:
            return i
        if right >= heap_len:
            if self._heap[i] < self._heap[left]:
                return left
            return i
        max_child = left
        if self._heap[left] < self._heap[right]:
            max_child = right
        if self._heap[i] < self._heap[max_child]:
            return max_child
        return i

    def _fix_down(self, i):
        local_max = self._get_local_max(i)
        if i < local_max:
            self._heap[i], self._heap[local_max] = (
                self._heap[local_max],
                self._heap[i],
            )
            return local_max
        return i

    def _fix_up(self, i):
        parent = _parent(i)
        if self._heap[parent] < self._heap[i]:
            self._heap[i], self._heap[parent] = self._heap[parent], self._heap[i]
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
        current = len(self._heap) - 1
        done = False
        while not done:
            local_max = self._fix_up(current)
            done = local_max == current
            current = local_max
