def partition(data, start, end, p, fun=lambda x: x):
    higher = start
    p_val = fun(p)
    for j in range(start, end):
        if fun(data[j]) < p_val:
            data[higher], data[j] = data[j], data[higher]
            higher += 1
    return higher

def partition_tuple(data, start, end, p):
    higher = start
    p_ord, _ = p
    for j in range(start, end):
        j_ord, _ = data[j]
        if j_ord < p_ord:
            data[higher], data[j] = data[j], data[higher]
            higher += 1
    return higher

def quickselect(data, start, end, k, fun=lambda x: x):
    start_, end_, higher = start, end, None
    while higher != k + 1:
        p = data[k]
        data[start_], data[k] = data[k], data[start_]
        higher = partition(data, start_ + 1, end_, p, fun)
        data[start_], data[higher - 1] = data[higher - 1], data[start_]
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher

def quickselect_tuple(data, start, end, k):
    start_, end_, higher = start, end, None
    while higher != k + 1:
        p = data[k]
        data[start_], data[k] = data[k], data[start_]
        higher = partition_tuple(data, start_ + 1, end_, p)
        data[start_], data[higher - 1] = data[higher - 1], data[start_]
        if k <= higher - 1:
            end_ = higher
        else:
            start_ = higher
