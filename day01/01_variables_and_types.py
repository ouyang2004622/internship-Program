# Python基础语法 - 变量和数据类型

# 1. 数字类型
integer_number = 42
float_number = 3.14
complex_number = 1 + 2j

print("整数:", integer_number)
print("浮点数:", float_number)
print("复数:", complex_number)

# 2. 字符串类型
string1 = "Hello, Python!"
string2 = '单引号字符串也可以'
multiline_string = """这是一个
多行字符串
示例"""

print("\n字符串示例:")
print(string1)
print(string2)
print(multiline_string)

# 3. 布尔类型
is_true = True
is_false = False

print("\n布尔值:")
print("True值:", is_true)
print("False值:", is_false)

# 4. 列表类型
my_list = [1, 2, "三", True]
print("\n列表示例:", my_list)
print("列表第一个元素:", my_list[0])

# 5. 元组类型
my_tuple = (1, "二", 3.0)
print("\n元组示例:", my_tuple)

# 6. 字典类型
my_dict = {
    "name": "张三",
    "age": 25,
    "city": "北京"
}
print("\n字典示例:", my_dict)
print("访问字典元素:", my_dict["name"])

# 7. 集合类型
my_set = {1, 2, 3, 3, 2, 1}  # 重复元素会自动去除
print("\n集合示例:", my_set) 