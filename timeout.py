import time
import multiprocessing

def slow_function():
    time.sleep(10)
    return "function finished"

def timeout_function(func, args=(), kwargs={}, timeout=5, default=None):
    with multiprocessing.Pool(processes=1) as pool:
        result = pool.apply_async(func, args=args, kwds=kwargs)
        try:
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            #print(f"function is over {timeout} seconds, timeout")
            return default