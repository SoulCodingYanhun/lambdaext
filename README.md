# Extended Lambda Library Documentation
# Extended Lambda 库文档

<div align="center">

![Extended Lambda Logo](https://via.placeholder.com/150?text=Ext`L)

*Enhancing Functional Programming in Python*  
*增强Python中的函数式编程*

</div>

---

## Table of Contents / 目录

1. [Introduction / 介绍](#introduction--介绍)
2. [Installation / 安装](#installation--安装)
3. [Core Concepts / 核心概念](#core-concepts--核心概念)
4. [API Reference / API参考](#api-reference--api参考)
5. [Advanced Usage / 高级用法](#advanced-usage--高级用法)
6. [Examples / 示例](#examples--示例)

---

## Introduction / 介绍

<table>
<tr>
<td width="50%">

The Extended Lambda Library is a powerful Python library that enhances functional programming capabilities in Python. It provides a set of tools and higher-order functions that allow for more expressive and concise functional code. This library extends the concept of lambda functions and introduces new ways to compose and manipulate functions.

</td>
<td width="50%">

Extended Lambda 库是一个强大的 Python 库,它增强了 Python 中的函数式编程能力。它提供了一套工具和高阶函数,允许更具表现力和简洁的函数式代码。这个库扩展了 lambda 函数的概念,并引入了组合和操作函数的新方法。

</td>
</tr>
</table>

---

## Installation / 安装

<table>
<tr>
<td width="50%">

To install the Extended Lambda Library, you can use pip:

```bash
pip install lambdaext
```

Or clone the repository and install it locally:

```bash
git clone https://github.com/SoulCodingYanhun/lambdaext.git
cd lambdaext
pip install -e .
```

</td>
<td width="50%">

要安装 Extended Lambda 库,你可以使用 pip:

```bash
pip install lambdaext
```

或者克隆仓库并本地安装:

```bash
git clone https://github.com/SoulCodingYanhun/lambdaext.git
cd lambdaext
pip install -e .
```

</td>
</tr>
</table>

---

## Core Concepts / 核心概念

<table>
<tr>
<th width="20%">Concept<br>概念</th>
<th width="40%">English</th>
<th width="40%">中文</th>
</tr>
<tr>
<td><code>LAMBDA</code></td>
<td>A special object that allows for the use of <code>=></code> and <code><=</code> like operators in Python.</td>
<td>一个特殊对象,允许在 Python 中使用类似 <code>=></code> 和 <code><=</code> 的运算符。</td>
</tr>
<tr>
<td><code>lambda_pass</code></td>
<td>Immediately executes a function and returns the result.</td>
<td>立即执行一个函数并返回结果。</td>
</tr>
<tr>
<td><code>lambda_return</code></td>
<td>Returns a callable <code>LambdaWrapper</code> object.</td>
<td>返回一个可调用的 <code>LambdaWrapper</code> 对象。</td>
</tr>
<tr>
<td>Composition<br>组合</td>
<td>Combining multiple functions into a single function.</td>
<td>将多个函数组合成一个单一函数。</td>
</tr>
<tr>
<td>Currying<br>柯里化</td>
<td>Transforming a function that takes multiple arguments into a sequence of functions, each taking a single argument.</td>
<td>将一个接受多个参数的函数转换为一系列函数,每个函数接受一个参数。</td>
</tr>
</table>

---

## API Reference / API参考

### Basic Operations / 基本操作

<table>
<tr>
<th width="30%">Operation<br>操作</th>
<th width="35%">English</th>
<th width="35%">中文</th>
</tr>
<tr>
<td><code>LAMBDA >> func</code></td>
<td>Equivalent to <code>lambda_pass(func)</code></td>
<td>等同于 <code>lambda_pass(func)</code></td>
</tr>
<tr>
<td><code>LAMBDA << func</code></td>
<td>Equivalent to <code>lambda_return(func)</code></td>
<td>等同于 <code>lambda_return(func)</code></td>
</tr>
</table>

### Higher-Order Functions / 高阶函数

<table>
<tr>
<th width="25%">Function<br>函数</th>
<th width="37.5%">English</th>
<th width="37.5%">中文</th>
</tr>
<tr>
<td><code>compose(*funcs)</code></td>
<td>Composes multiple functions from right to left.</td>
<td>从右到左组合多个函数。</td>
</tr>
<tr>
<td><code>pipe(*funcs)</code></td>
<td>Composes multiple functions from left to right.</td>
<td>从左到右组合多个函数。</td>
</tr>
<tr>
<td><code>curry(func)</code></td>
<td>Returns a curried version of the function.</td>
<td>返回函数的柯里化版本。</td>
</tr>
<tr>
<td><code>map_func(func)</code></td>
<td>Returns a new function that applies <code>func</code> to each element of an iterable.</td>
<td>返回一个新函数,该函数将 <code>func</code> 应用于可迭代对象的每个元素。</td>
</tr>
<tr>
<td><code>filter_func(predicate)</code></td>
<td>Returns a new function that filters an iterable using the given predicate.</td>
<td>返回一个新函数,该函数使用给定的谓词过滤可迭代对象。</td>
</tr>
<tr>
<td><code>reduce_func(func)</code></td>
<td>Returns a new function that reduces an iterable using the given function.</td>
<td>返回一个新函数,该函数使用给定的函数归约可迭代对象。</td>
</tr>
</table>

### Advanced Functions / 高级函数

<table>
<tr>
<th width="25%">Function<br>函数</th>
<th width="37.5%">English</th>
<th width="37.5%">中文</th>
</tr>
<tr>
<td><code>keyword_params_func(func)</code></td>
<td>Handles functions that only accept keyword arguments.</td>
<td>处理只接受关键字参数的函数。</td>
</tr>
<tr>
<td><code>combined_params_func(func)</code></td>
<td>Handles functions that accept positional, variable, and keyword arguments.</td>
<td>处理接受位置参数、可变参数和关键字参数的函数。</td>
</tr>
<tr>
<td><code>conditional_func(condition, true_func, false_func)</code></td>
<td>Implements conditional expression functionality.</td>
<td>实现条件表达式功能。</td>
</tr>
<tr>
<td><code>list_comp_func(condition)</code></td>
<td>Implements list comprehension functionality.</td>
<td>实现列表推导式功能。</td>
</tr>
<tr>
<td><code>dict_comp_func(condition)</code></td>
<td>Implements dictionary comprehension functionality.</td>
<td>实现字典推导式功能。</td>
</tr>
<tr>
<td><code>nested_func(outer_func, inner_func)</code></td>
<td>Creates nested functions.</td>
<td>创建嵌套函数。</td>
</tr>
<tr>
<td><code>immediate_func(func, arg)</code></td>
<td>Immediately calls the given function and returns the result.</td>
<td>立即调用给定函数并返回结果。</td>
</tr>
</table>

---

## Advanced Usage / 高级用法

<table>
<tr>
<td width="50%">

The Extended Lambda Library allows for complex function compositions and manipulations. Here are some advanced usage patterns:

1. **Function Composition**
2. **Currying**
3. **Conditional Lambda**

</td>
<td width="50%">

Extended Lambda 库允许复杂的函数组合和操作。以下是一些高级用法模式:

1. **函数组合**
2. **柯里化**
3. **条件 Lambda**

</td>
</tr>
</table>

---

## Examples / 示例

### 1. Using LAMBDA operators / 使用 LAMBDA 运算符

<table>
<tr>
<td width="50%">

```python
from lambdaext import LAMBDA

double = LAMBDA >> (lambda x: x * 2)
triple = LAMBDA << (lambda x: x * 3)

print(double(5))    # Output: 10
print(triple(5)())  # Output: 15
```

</td>
<td width="50%">

```python
from lambdaext import LAMBDA

double = LAMBDA >> (lambda x: x * 2)
triple = LAMBDA << (lambda x: x * 3)

print(double(5))    # 输出: 10
print(triple(5)())  # 输出: 15
```

</td>
</tr>
</table>

### 2. List manipulation / 列表操作

<table>
<tr>
<td width="50%">

```python
from lambdaext import map_func, filter_func, reduce_func

numbers = [1, 2, 3, 4, 5]
doubled = list(map_func(lambda x: x * 2)(numbers))
evens = list(filter_func(lambda x: x % 2 == 0)(doubled))
total = reduce_func(lambda x, y: x + y)(evens)

print(doubled)  # Output: [2, 4, 6, 8, 10]
print(evens)    # Output: [2, 4, 6, 8, 10]
print(total)    # Output: 30
```

</td>
<td width="50%">

```python
from lambdaext import map_func, filter_func, reduce_func

numbers = [1, 2, 3, 4, 5]
doubled = list(map_func(lambda x: x * 2)(numbers))
evens = list(filter_func(lambda x: x % 2 == 0)(doubled))
total = reduce_func(lambda x, y: x + y)(evens)

print(doubled)  # 输出: [2, 4, 6, 8, 10]
print(evens)    # 输出: [2, 4, 6, 8, 10]
print(total)    # 输出: 30
```

</td>
</tr>
</table>

### 3. Complex composition / 复杂组合

<table>
<tr>
<td width="50%">

```python
from lambdaext import compose, curry, map_func

@curry
def add(x, y):
    return x + y

increment = add(1)
double = lambda x: x * 2

process = compose(
    list,
    map_func(double),
    map_func(increment)
)

result = process([1, 2, 3, 4, 5])
print(result)  # Output: [4, 6, 8, 10, 12]
```

</td>
<td width="50%">

```python
from lambdaext import compose, curry, map_func

@curry
def add(x, y):
    return x + y

increment = add(1)
double = lambda x: x * 2

process = compose(
    list,
    map_func(double),
    map_func(increment)
)

result = process([1, 2, 3, 4, 5])
print(result)  # 输出: [4, 6, 8, 10, 12]
```

</td>
</tr>
</table>

---

<div align="center">

**Extended Lambda Library: Elevating Functional Programming in Python**  
**Extended Lambda 库: 提升 Python 中的函数式编程**

[GitHub](https://github.com/SoulCodingYanhun/lambdaext) | [PyPI](https://pypi.org/project/lambdaext) | [CodeStarLabs](https://codestarlabs.top)

</div>