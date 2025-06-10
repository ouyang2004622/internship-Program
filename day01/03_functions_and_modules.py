# Python基础语法 - 函数和模块

# 1. 基本函数定义和调
def greet(name):
    """这是一个简单的问候函数"""
    return f"你好, {name}!"

print("基本函数示例:")
print(greet("小明"))

# 2. 带默认参数的函数
def power(x, n=2):
    """计算x的n次方"""
    return x ** n

print("\n默认参数函数示例:")
print("2的平方:", power(2))      # 使用默认参数
print("2的3次方:", power(2, 3))  # 指定参数

# 3. 多个返回值的函数
def calculate(a, b):
    """返回两个数的和与差"""
    return a + b, a - b

print("\n多返回值函数示例:")
sum_result, diff_result = calculate(10, 5)
print("和:", sum_result)
print("差:", diff_result)

# 4. 可变参数函数
def sum_numbers(*args):
    """计算所有传入数字的和"""
    return sum(args)

print("\n可变参数函数示例:")
print("1 + 2 + 3 =", sum_numbers(1, 2, 3))
print("1 + 2 + 3 + 4 =", sum_numbers(1, 2, 3, 4))

# 5. 关键字参数函数
def print_info(**kwargs):
    """打印所有传入的关键字参数"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print("\n关键字参数函数示例:")
print_info(name="张三", age=25, city="北京")

# 6. 匿名函数（lambda）
square = lambda x: x ** 2
print("\n匿名函数示例:")
print("5的平方:", square(5))

# 7. 模块导入示例
import math
import random
from datetime import datetime

print("\n模块使用示例:")
print("π的值:", math.pi)
print("随机数(1-10):", random.randint(1, 10))
print("当前时间:", datetime.now()) 