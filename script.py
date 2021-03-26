import time


file = open("words_alpha.txt", "r")
t = time.time()

# print(len(file.readlines()))

count = 0
index = 0
maxLine = 0
for line in file.readlines():

    if (len(line) > maxLine):
        maxLine = len(line)

    for char in line:
        if char == "a" and index < 200000:
            count += 1
    index += 1


print(count)


file.close()
