def checkNonNegativeArgs(index):
    def nonNegative(f):
        def wrap(*args):
            if(args[index]<0):
                raise ValueError('index {}  is negative'.format(index))
            return f
        return wrap
    return nonNegative

@checkNonNegativeArgs(2)
def test(*args):
    print(args)
