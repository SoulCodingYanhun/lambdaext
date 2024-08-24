from .core import lambda_pass, lambda_return

def pass_lambda(func):
    """
    装饰器版本的 =>
    立即执行函数并返回结果
    """
    return lambda_pass(func)

def return_lambda(func):
    """
    装饰器版本的 <=
    返回可调用的 LambdaWrapper 对象
    """
    return lambda_return(func)