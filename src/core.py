import functools
from typing import Callable, Iterable, Any, Dict, List, Tuple

class LambdaWrapper:
    def __init__(self, func):
        self.func = func
        self.args = ()
        self.kwargs = {}

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            return self.func(*self.args, **self.kwargs)
        else:
            self.args = args
            self.kwargs = kwargs
            return self

def lambda_pass(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def lambda_return(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return LambdaWrapper(func)(*args, **kwargs)
    return wrapper

class LambdaBuilder:
    def __rshift__(self, func):
        return lambda_pass(func)

    def __lshift__(self, func):
        return lambda_return(func)

LAMBDA = LambdaBuilder()

# 基本高阶函数
def compose(*funcs):
    """从右到左组合多个函数"""
    def compose_two(f, g):
        return lambda x: f(g(x))
    return functools.reduce(compose_two, funcs, lambda x: x)

def pipe(*funcs):
    """从左到右组合多个函数"""
    return compose(*reversed(funcs))

def curry(func):
    """函数柯里化"""
    @functools.wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return lambda *more_args, **more_kwargs: curried(*(args + more_args), **{**kwargs, **more_kwargs})
    return curried

# 高阶函数扩展
def map_func(func: Callable) -> Callable[[Iterable], map]:
    """返回一个新函数，该函数将原函数应用于可迭代对象的每个元素"""
    return lambda iterable: map(func, iterable)

def filter_func(predicate: Callable) -> Callable[[Iterable], filter]:
    """返回一个新函数，该函数使用给定的谓词函数过滤可迭代对象"""
    return lambda iterable: filter(predicate, iterable)

def reduce_func(func: Callable) -> Callable[[Iterable, Any], Any]:
    """返回一个新函数，该函数使用给定的函数对可迭代对象进行归约"""
    def reducer(iterable, initial=None):
        if initial is None:
            return functools.reduce(func, iterable)
        return functools.reduce(func, iterable, initial)
    return reducer

def partial_func(func: Callable, *args, **kwargs) -> Callable:
    """返回一个新函数，该函数是原函数的部分应用"""
    return functools.partial(func, *args, **kwargs)

def memoize(func: Callable) -> Callable:
    """记忆化装饰器，缓存函数的结果"""
    cache = {}
    @functools.wraps(func)
    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return memoized

def compose_predicates(*predicates):
    """组合多个谓词函数，返回一个新的谓词函数"""
    return lambda x: all(p(x) for p in predicates)

def flip(func: Callable) -> Callable:
    """返回一个新函数，该函数将原函数的前两个参数顺序颠倒"""
    @functools.wraps(func)
    def flipped(a, b, *args, **kwargs):
        return func(b, a, *args, **kwargs)
    return flipped

def identity(x: Any) -> Any:
    """恒等函数"""
    return x

def keyword_params_func(func: Callable) -> Callable[..., Any]:
    """
    返回一个新函数，该函数接受关键字可变参数
    """
    @functools.wraps(func)
    def wrapper(**kwargs):
        return func(**kwargs)
    return wrapper

def combined_params_func(func: Callable) -> Callable[..., Any]:
    """
    返回一个新函数，该函数结合了位置参数、可变参数和关键字可变参数
    """
    @functools.wraps(func)
    def wrapper(x, *args, **kwargs):
        return func(x, *args, **kwargs)
    return wrapper

def conditional_func(condition: Callable[[Any], bool], true_func: Callable, false_func: Callable) -> Callable:
    """
    返回一个新函数，根据条件选择执行true_func或false_func
    """
    return lambda x: true_func(x) if condition(x) else false_func(x)

def list_comp_func(condition: Callable[[Any], bool]) -> Callable[[Iterable], List]:
    """
    返回一个新函数，该函数使用列表推导式和给定的条件
    """
    return lambda x: [i for i in x if condition(i)]

def dict_comp_func(condition: Callable[[Any, Any], bool]) -> Callable[[Dict], Dict]:
    """
    返回一个新函数，该函数使用字典推导式和给定的条件
    """
    return lambda x: {k: v for k, v in x.items() if condition(k, v)}

def nested_func(outer_func: Callable, inner_func: Callable) -> Callable:
    """
    返回一个嵌套函数，outer_func包含inner_func
    """
    return lambda x: outer_func(inner_func(x))

def immediate_func(func: Callable, arg: Any) -> Any:
    """
    立即调用给定的函数并返回结果
    """
    return func(arg)

# 更新 __all__ 列表
__all__ = [
    'LAMBDA', 'lambda_pass', 'lambda_return', 'compose', 'pipe', 'curry',
    'map_func', 'filter_func', 'reduce_func', 'partial_func', 'memoize',
    'compose_predicates', 'flip', 'identity',
    'keyword_params_func', 'combined_params_func', 'conditional_func',
    'list_comp_func', 'dict_comp_func', 'nested_func', 'immediate_func'
]