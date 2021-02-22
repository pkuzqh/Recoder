import os
from tqdm import tqdm
import pickle
lst = ["train", "dev"]
rules = {"pad":0}
onelist = ["list"]
rulelist = []
fatherlist = []
fathername = []
depthlist = []
copynode = []
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
def parseTree(treestr):
    tokens = treestr.split()
    root = Node("root", 0)
    currnode = root
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            nnode = Node(x, i + 1)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
        else:
            currnode = currnode.father
    return root
def getRule(node, nls, currId, d):
    global rules
    global onelist
    global rulelist
    global fatherlist
    global depthlist
    global copynode
    if len(node.child) == 0:
        return [], []
        if " -> End " not in rules:
            rules[" -> End "] = len(rules)
        return [rules[" -> End "]]
    child = sorted(node.child, key=lambda x:x.name)
    if len(node.child) == 1 and node.child[0].name in nls:
        copynode.append(node.name)
        rulelist.append(10000 + nls.index(node.child[0].name))
        fatherlist.append(currId)
        fathername.append(node.name)
        depthlist.append(d)
        currid = len(rulelist) - 1
        for x in child:
            getRule(x, nls, currId, d + 1)
            #rulelist.extend(a)
            #fatherlist.extend(b)
    else:
        if node.name not in onelist:
            rule = node.name + " -> "
            for x in child:
                rule += x.name + " "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)
            currid = len(rulelist) - 1
            for x in child:
                getRule(x, nls, currid, d + 1)
        else:
            for x in (child):
                rule = node.name + " -> " + x.name
                if rule in rules:
                    rulelist.append(rules[rule])
                else:
                    rules[rule] = len(rules)
                    rulelist.append(rules[rule])
                fatherlist.append(currId)
                fathername.append(node.name)
                depthlist.append(d)
                getRule(x, nls, len(rulelist) - 1, d + 1)
            rule = node.name + " -> End "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
            depthlist.append(d)
    '''if node.name == "root":
        print('rr')
        print('rr')
        print(rulelist)'''
    '''rule = " -> End "
    if rule in rules:
        rulelist.append(rules[rule])
    else:
        rules[rule] = len(rules)
        rulelist.append(rules[rule])'''
    #return rulelist, fatherlist
for x in lst:
    inputdir = x + "_input/"
    outputdir = x + "_output/"
    wf = open(x + ".txt", "w")
    for i in tqdm(range(len(os.listdir(inputdir)))):
        fname = inputdir + str(i + 1) + ".txt"
        ofname = outputdir + str(i + 1) + ".txt"
        f = open(fname, "r")
        nls = f.read()
        f.close()
        f = open(ofname, "r")
        asts = f.read()
        f.close()
        wf.write(nls + "\n")
        #wf.write(asts + "\n")
        assert(len(asts.split()) == 2 * asts.split().count('^'))
        root = parseTree(asts)
        rulelist = []
        fatherlist = []
        fathername = []
        depthlist = []
        getRule(root, nls.split(), -1, 2)
        s = ""
        for x in rulelist:
            s += str(x) + " "
        wf.write(s + "\n")
        s = ""
        for x in fatherlist:
            s += str(x) + " "
        wf.write(s + "\n")
        s = ""
        for x in depthlist:
            s += str(x) + " "
        wf.write(s + "\n")
        wf.write(" ".join(fathername) + "\n")
        #print(rules)
        #print(asts)
wf.close()
wf = open("rule.pkl", "wb")
wf.write(pickle.dumps(rules))
wf.close()
print(copynode)