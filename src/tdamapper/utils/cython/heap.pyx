cdef int _left(int i):
    return 2 * i + 1


cdef int _right(int i):
    return 2 * i + 2


cdef int _parent(int i):
    return max(0, (i - 1) // 2)


cdef class _HeapNode:
    cdef object __key
    cdef object __value

    def __init__(self, object key, object value):
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


cdef class MaxHeap:

    def __init__(self):
        self.__heap = []

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
            return None
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

    cdef int _get_local_max(self, int i):
        cdef int heap_len = len(self.__heap)
        cdef int left = _left(i)
        cdef int right = _right(i)
        if left >= heap_len:
            return i
        if right >= heap_len:
            if self.__heap[i] < self.__heap[left]:
                return left
            return i
        cdef int max_child = left
        if self.__heap[left] < self.__heap[right]:
            max_child = right
        if self.__heap[i] < self.__heap[max_child]:
            return max_child
        return i

    cdef int _fix_down(self, int i):
        cdef int local_max = self._get_local_max(i)
        if i < local_max:
            self.__heap[i], self.__heap[local_max] = self.__heap[local_max], self.__heap[i]
            return local_max
        return i

    cdef int _fix_up(self, int i):
        cdef int parent = _parent(i)
        if self.__heap[parent] < self.__heap[i]:
            self.__heap[i], self.__heap[parent] = self.__heap[parent], self.__heap[i]
            return parent
        return i

    cdef void _bubble_down(self):
        cdef int current = 0
        cdef bint done = False
        cdef int local_max
        while not done:
            local_max = self._fix_down(current)
            done = current == local_max
            current = local_max

    cdef void _bubble_up(self):
        cdef int current = len(self.__heap) - 1
        cdef bint done = False
        cdef int local_max
        while not done:
            local_max = self._fix_up(current)
            done = local_max == current
            current = local_max
