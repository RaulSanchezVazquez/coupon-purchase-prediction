#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:05:42 2018

@author: raulsanchez
"""

import multiprocessing
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    """
    Thanks to:
    stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
    
    The multiprocessing.pool.Pool class creates the worker processes in 
    its __init__ method, makes them daemonic and starts them, 
    and it is not possible to re-set their daemon attribute to False before 
    they are started (and afterwards it's not allowed anymore). 
    But you can create your own sub-class of 
    multiprocesing.pool.Pool (multiprocessing.Pool is just a wrapper function) 
    and substitute your own multiprocessing.Process sub-class, 
    which is always non-daemonic, to be used for the worker processes.
    """
    def _get_daemon(self):
        """ get daemon """
        return False
    def _set_daemon(self, value):
        """ set daemon """
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    """ 
    This is a custom pool. It is used to allow the apply() method
    to nest other other pools.
    """
    
    Process = NoDaemonProcess

def apply(func, data, n_jobs=8):
    """ 
    Map a parallel function to a list 
    
    Params:
    ---------
    func : function The function to be pararellized
    data : list Items in which to apply func
    n_jobs : int
    
    Return:
    ---------
    list : Results in the same order as found in data
    
    Example:
    ---------
    
    import parallel
    
    #Simple example
    mylist = range(10)
    
    def power_of_two(x):
        return x**2
        
    parallel.apply(
        power_of_two,
        mylist)
    
    Out:
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    
    """

    pool = MyPool(n_jobs)
    
    result = pool.map(
        func,
        data)
    
    pool.close()
    pool.join()
    return result