class MaxHeap:

    def __init__(self):
        self.__heap = []

    def __iter__(self):
        return iter(self.__heap)

    def __next__(self):
        return next(self.__heap)

    def __len__(self):
        return len(self.__heap)

    def top(self):
        if not self.__heap:
            return None
        return self.__heap[0]

    def pop(self):
        if not self.__heap:
            return
        last = self.__heap.pop()
        max_val = self.__heap[0]
        self.__heap[0] = last
        self._bubble_down()
        return max_val

    def add(self, v):
        self.__heap.append(v)
        self._bubble_up()

    def _get_local_max(self, i):
        heap_len = len(self.__heap)
        left = self._left(i)
        right = self._right(i)
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

    def _fix_local_max(self, i):
        local_max = self._get_local_max(i)
        if i < local_max:
            self.__heap[i], self.__heap[local_max] = self.__heap[local_max], self.__heap[i]
            return local_max
        return i

    def _bubble_down(self):
        current = 0
        done = False
        while not done:
            local_max = self._fix_local_max(current)
            done = current == local_max
            current = local_max

    def _bubble_up(self):
        current = len(self.__heap) - 1
        done = False
        while not done:
            parent = self._parent(current)
            local_max = self._fix_local_max(parent)
            done = local_max == parent
            current = parent

    def _left(self, i):
        return 2 * i + 1

    def _right(self, i):
        return 2 * i + 2

    def _parent(self, i):
        return max(0, (i - 1) // 2)
