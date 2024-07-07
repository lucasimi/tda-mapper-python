def __swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def partition(data, start, end, p_ord):
    higher = start
    for j in range(start, end):
        j_ord, _ = data[j]
        if j_ord < p_ord:
            __swap(data, higher, j)
            higher += 1
    return higher


def quickselect(data, start, end, k):
    if (k < start) or (k >= end):
        return
    start_, end_, higher = start, end, None
    while higher != k + 1:
        p, _ = data[k]
        __swap(data, start_, k)
        higher = partition(data, start_ + 1, end_, p)
        __swap(data, start_, higher - 1)
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher


def partition_tuple(data_ord, data_arr, start, end, p_ord):
    higher = start
    for j in range(start, end):
        j_ord = data_ord[j]
        if j_ord < p_ord:
            __swap(data_arr, higher, j)
            __swap(data_ord, higher, j)
            higher += 1
    return higher


def quickselect_tuple(data_ord, data_arr, start, end, k):
    if (k < start) or (k >= end):
        return
    start_, end_, higher = start, end, None
    while higher != k + 1:
        p_ord = data_ord[k]
        __swap(data_arr, start_, k)
        __swap(data_ord, start_, k)
        higher = partition_tuple(data_ord, data_arr, start_ + 1, end_, p_ord)
        __swap(data_arr, start_, higher - 1)
        __swap(data_ord, start_, higher - 1)
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher
