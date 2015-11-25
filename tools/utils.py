import bz2
import sys

def openr( fn, mode = "r" ):
    if fn is None:
        return sys.stdin
    return bz2.BZ2File(fn) if fn.endswith(".bz2") else open(fn,mode)

def openw( fn ):
    if fn is None:
        return sys.stdout
    return bz2.BZ2File(fn,"w") if fn.endswith(".bz2") else open(fn,"w")



def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False