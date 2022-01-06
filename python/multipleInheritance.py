class Base:
    def __init__(self):
        print('Base init')

    def f(self):
        print('base f()')

class sub(Base):

    def f(self):
        print('sub f()')

class sub2(Base):
    def __init__(self):
        print('sub2 init')


class SimpleList:

    def __init__(self,items):
        self._items=list(items)

    def add(self,item):
        self._items.apend(item)

    def sort(self):
        return self._items.sort()

    def __getitem__(self, index):
        return self._items[index]

    def __repr__(self):
        print(self._items)

class SortedList(SimpleList):

    def __init__(self,items=()):
        self._items=list(items)
        self.sort()

    def add(self,item):
        self._items.apend(item)
        self.sort()

class IntList(SimpleList):
    def __init__(self,items=()):
       for x in items:  self._validate_(x)
       super().__init__(items)

    @staticmethod
    def _validate_(item):
       if not  isinstance(item,int):
           raise TypeError('the type if the item is not int')

    def add(self,item):
        self._validate_(item)
        self.add(item)

class SortedIntList(IntList,SortedList):
    pass




