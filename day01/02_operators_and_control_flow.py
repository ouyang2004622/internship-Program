# Python基础语法 - 运算符和控制流

# 1. 算术运算符
print("算术运算符示例:")
a = 10
b = 3
print("加法:", a + b)
print("减法:", a - b)
print("乘法:", a * b)
print("除法:", a / b)
print("整除:", a // b)
print("取余:", a % b)
print("幂运算:", a ** b)

# 2. 比较运算符
print("\n比较运算符示例:")
x = 5
y = 10
print("x =", x, "y =", y)
print("x > y:", x > y)
print("x < y:", x < y)
print("x == y:", x == y)
print("x != y:", x != y)
print("x >= y:", x >= y)
print("x <= y:", x <= y)

# 3. 逻辑运算符
print("\n逻辑运算符示例:")
p = True
q = False
print("p and q:", p and q)
print("p or q:", p or q)
print("not p:", not p)

# 4. if-elif-else条件语句
print("\nif-elif-else条件语句示例:")
score = 85

if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")

# 5. for循环
print("\nfor循环示例:")
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(fruit)

# 使用range()
print("\n使用range()的for循环:")
for i in range(3):
    print(f"计数: {i}")

# 6. while循环
print("\nwhile循环示例:")
count = 0
while count < 3:
    print(f"当前count值: {count}")
    count += 1

# 7. break和continue语句
print("\nbreak和continue示例:")
for i in range(5):
    if i == 2:
        continue  # 跳过2
    if i == 4:
        break    # 到4时退出循环
    print(f"数字: {i}") 