class callCount:
    def __init__(self,f):
        self.f=f
        self.callcount=0

    def __call__(self,*args,**kwargs):
        self.callcount+=1
        return self.f(*args,**kwargs)

@callCount
def hello(name):
    print('hello {}'.format(name))

@callCount
def hellome(name):
    print('hello {}')