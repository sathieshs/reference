def escape_unicode(f):
    def wrap(*args,**kwargs):
        x=f(*args,**kwargs)
        return x+' 123'
    return wrap

@escape_unicode
def northern_city():
    return 'dgkfsnjdhjg'



f=northern_city()
print(f)