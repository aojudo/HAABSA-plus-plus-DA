# repeats contents of a text file n times

n = 3.5

with open('test.txt', 'r+') as file:
    text = file.read()
    file.write(text * (n-1))