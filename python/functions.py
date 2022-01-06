def sortByLastLetter(strings):
    def lastLetter(letter):
        return letter[-1]
    return sorted(strings,key=lastLetter)

g='global'
def outer(p='param'):
    l='local'
    def inner():
        print(g,p,l)
    inner()


def raiseTothePower(y=0):
    def x_to_the_power_y(x=1):
        return x**y
    return x_to_the_power_y



