import time

def timeit(method):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper
