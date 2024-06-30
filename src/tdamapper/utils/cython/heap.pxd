cdef class MaxHeap:

    cdef list __heap
    cdef object __iter

    cdef int _get_local_max(self, int i)

    cdef int _fix_down(self, int i)

    cdef int _fix_up(self, int i)

    cdef void _bubble_down(self)

    cdef void _bubble_up(self)
