import contextlib
import sys

@contextlib.contextmanager
def logging_context_manager():
    print('__enter')
    try:
        yield 'you atr in a with block  __return value of __enter__'
        print('normal exit')
    except Exception:
        print('exception exit')



