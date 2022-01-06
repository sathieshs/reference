import time


def timer():
    last_called = None

    def elapsed():
        nonlocal last_called
        now = time.time()
        if last_called is None:
            last_called = now
            return None
        elapsed = now - last_called
        last_called = now
        print(elapsed)
        return elapsed
    return elapsed

