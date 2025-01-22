##########################ABSTRACTION
from abs import ABC, abstractmethod
class Computer(ABC):
  @abstractmethod
  def processs(self):
    pass
classs Laptop(Computer):
  def process(self): # -> abs methods must be defind in base class
    print("running")
#com = Computer()  -> Can't make object of abs class
com1= laptop()

########################## Decotrator
def div(a, b)
    print( a/b)
def smart_div(func):
  def inner(a,b):
    if a>b:
      a,b = b,a
    return inner
div = smart_div(div)
div(a,b)
############## LIST COMPREHENSION
# Online Python - IDE, Editor, Compiler, Interpreter
a = [1,4,5,65,67,7]
print([x**2 for x in a if 4 < x < 10]) #[25, 49]
print([x if x > 10 else 0 for x in a]) #[0, 0, 0, 65, 67, 0]
a = [['4', '8'], ['4', '2', '28'], ['1', '12'], ['3', '6', '2']] 
print([int(x) for inlist in a for x in inlist]) #[4, 8, 4, 2, 28, 1, 12, 3, 6, 2]
print([[int(x) for x in inlist] for inlist in a]) #[[4, 8], [4, 2, 28], [1, 12], [3, 6, 2]]

#######STATIC method not receive an implicit first argument. This method can’t access or modify the class state
class C(object):
    @staticmethod
    def fun(arg1, arg2, ...):
        ...
returns: a static method for function fun.
#######STATIC classs decorator is a built-in function decorator that is an expression that gets evaluated after your function is defined. 
class C(object):
    school = " samsung"
    @classmethod
    def fun(cls):
        print(cls.school)
C.fun()

