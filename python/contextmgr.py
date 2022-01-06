class LoggingContextManager:

    def __enter__(self):
        print('entering')
        return 'you are in with block';

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exiting {} {} {}'.format(exc_type, exc_val, exc_tb))

        pass


