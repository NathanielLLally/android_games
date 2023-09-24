l = [1,2,3]
l.insert(0,0)
print(l)
for x in range(10,20,1):
    l.insert(0,x)
    if (len(l) >= 10):
        l.pop()
    print(l)
