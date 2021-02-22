lines1 = open("location2/JacksonDatabind/16/parsed_ochiai_result", "r").readlines()
lines2 = open("location2/JacksonDatabind/2/parsed_ochiai_result", "r").readlines()
d = []
for x in lines2:
    d.append(x.split("#")[0])
for i in range(len(lines1)):
    lst1 = lines1[i].strip().split()[0].split("#")[0]
    lst2 = lines2[i].strip().split()[0].split("#")[0]
    if lst1 not in d[:300] and "Exception" not in lst1:
        print(lst1)
print(lst1, lst2, i)