"""
    Multiprocessing
"""
from typing import Callable, Iterable
from multiprocessing.pool import ThreadPool, AsyncResult
from collections import deque


def async_map(map_fn: Callable, iterable: Iterable, num_parallel_calls: int = 32):
    iterator = iter(iterable)

    tasks = deque()  # type: deque[AsyncResult]
    ending = False

    with ThreadPool() as pool:
        while len(tasks) != 0 or not ending:
            while not ending and len(tasks) < num_parallel_calls:
                try:
                    elem = next(iterator)
                    task = pool.apply_async(map_fn, args=(elem,))
                    tasks.append(task)
                except StopIteration:
                    ending = True

            yield tasks.popleft().get()
