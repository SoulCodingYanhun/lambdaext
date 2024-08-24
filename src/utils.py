from .core import LAMBDA

def compose(*funcs):
    """
    函数组合工具
    从右到左组合多个函数
    """
    def compose_two(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose_two, funcs, lambda x: x)

def pipe(*funcs):
    """
    函数管道工具
    从左到右组合多个函数
    """
    return compose(*reversed(funcs))

def curry(func):
    """
    函数柯里化工具
    """
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})
    return curried