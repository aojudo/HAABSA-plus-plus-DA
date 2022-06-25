
str1 = 'hi-sir'
str2 = 'hi'



def split_sir(str):
    str = str.split('-')
    first = str[0]
    is_sir = False
    if len(str) > 1:
        is_sir = True
    
    return first, is_sir


print(str(split_sir(str1)))
print(str(split_sir(str2)))
