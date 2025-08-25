def timer(func):
    """
    Wrapper function to time the execution of a function.
    """
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.8f} seconds", flush=True)
        return result
    return wrapper
