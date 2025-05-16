from typing import List
import time

def knap_dp_bottom_up(v: List[int], w: List[int], W: int) -> int:
    n = len(v)
    M = [[0] * (W + 1) for _ in range(n + 1)]
    for j in range(1, n + 1):
        pj = w[j - 1]
        vj = v[j - 1]
        for X in range(W + 1):
            if pj > X:
                M[j][X] = M[j - 1][X]
            else:
                M[j][X] = max(vj + M[j - 1][X - pj], M[j - 1][X])
    return M[n][W]

def knap_bf_recursive(n: int, v: List[int], w: List[int], W: int) -> int:
    if n == 0 or W == 0:
        return 0
    if w[n - 1] > W:
        return knap_bf_recursive(n - 1, v, w, W)
    use = v[n - 1] + knap_bf_recursive(n - 1, v, w, W - w[n - 1])
    dont_use = knap_bf_recursive(n - 1, v, w, W)
    return use if use > dont_use else dont_use

def timed_bf(v, w, W):
    start = time.time()
    opt = knap_bf_recursive(len(v), v, w, W)
    end = time.time()
    return opt, end - start


def timed_knap(v, w, W):
    start = time.time()
    opt = knap_dp_bottom_up(v, w, W)
    end = time.time()
    return opt, end - start