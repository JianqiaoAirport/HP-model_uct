def word_to_first_letter(str):
    lst = ""
    str.split("\', \'")

str = "'START_TOKEN', 'UP', 'RIGHT', 'BACKWARD', 'LEFT', 'UP', 'UP', 'RIGHT', 'DOWN', 'FORWARD', 'LEFT', 'UP', 'LEFT', 'DOWN', 'BACKWARD', 'DOWN', 'LEFT', 'FORWARD', 'RIGHT', 'DOWN', 'LEFT', 'BACKWARD', 'RIGHT', 'RIGHT', 'DOWN', 'RIGHT', 'UP', 'RIGHT', 'FORWARD', 'LEFT', 'FORWARD', 'RIGHT', 'UP', 'LEFT', 'LEFT', 'FORWARD', 'DOWN', 'BACKWARD', 'DOWN', 'LEFT', 'UP', 'UP', 'FORWARD', 'UP', 'BACKWARD', 'RIGHT', 'RIGHT', 'UP', 'BACKWARD', 'RIGHT', 'RIGHT', 'BACKWARD', 'LEFT', 'BACKWARD', 'LEFT', 'DOWN', 'RIGHT', 'FORWARD', 'DOWN', 'BACKWARD', 'LEFT', 'LEFT', 'UP', 'UP'"
lst = str.split("', '")
lst = lst[1:]
result = ""
for i in lst:
    result += i[0]

print(result)