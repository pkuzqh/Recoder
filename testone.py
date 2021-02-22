import os
import javalang
#from ast import nodes
from graphviz import Digraph
import json
import pickle
from tqdm import tqdm
import numpy as np
from run import *
from stringfycode import stringfyRoot
from copy import deepcopy
import time
import io
import subprocess
from Searchnode import Node
import traceback
linenode = ['Statement_ter', 'BreakStatement_ter', 'ReturnStatement_ter', 'ContinueStatement', 'ContinueStatement_ter', 'LocalVariableDeclaration', 'condition', 'control', 'BreakStatement', 'ContinueStatement', 'ReturnStatement', "parameters", 'StatementExpression', 'return_type']
#os.environ["CUDA_VISIBLE_DEVICES"]="1, 4"
def getLocVar(node):
  varnames = []
  if node.name == 'VariableDeclarator':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'FormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'InferredFormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  for x in node.child:
    varnames.extend(getLocVar(x))
  return varnames
n = 0
def setid(root):
  global n
  root.id = n
  n += 1
  for x in root.child:
    setid(x)
def solveLongTree(root, subroot):
    global n
    m = 'None'
    troot = 'None'
    for x in root.child:
        if x.name == 'name':
            m = x.child[0].name
    if len(root.getTreestr().strip().split()) >= 1000:
        tmp = subroot
        if len(tmp.getTreestr().split()) >= 1000:
            assert(0)
        lasttmp = None
        while True:
            if len(tmp.getTreestr().split()) >= 1000:
                break
            lasttmp = tmp
            tmp = tmp.father
        index = tmp.child.index(lasttmp)
        ansroot = Node(tmp.name, 0)
        ansroot.child.append(lasttmp)
        ansroot.num = 2 + len(lasttmp.getTreestr().strip().split())
        while True:
            b = True
            afternode = tmp.child.index(ansroot.child[-1]) + 1
            if afternode < len(tmp.child) and ansroot.num + tmp.child[afternode].getNum() < 1000:
                b = False
                ansroot.child.append(tmp.child[afternode])
                ansroot.num += tmp.child[afternode].getNum()
            prenode = tmp.child.index(ansroot.child[0]) - 1
            if prenode >= 0 and ansroot.num + tmp.child[prenode].getNum() < 1000:
                b = False
                ansroot.child.append(tmp.child[prenode])
                ansroot.num += tmp.child[prenode].getNum()
            if b:
                break
        troot = ansroot
    else:
        troot = root
    n = 0
    setid(troot)
    varnames = getLocVar(troot)
    fnum = -1
    vnum = -1
    vardic = {}
    vardic[m] = 'meth0'
    typedic = {}
    for x in varnames:
        if x[1].name == 'VariableDeclarator':
            vnum += 1
            vardic[x[0]] = 'loc' + str(vnum)
            t = -1
            for s in x[1].father.father.child:
                #print(s.name)
                if s.name == 'type':
                    t = s.child[0].child[0].child[0].name[:-4]
                    break
            assert(t != -1)
            typedic[x[0]] = t
        else:
            fnum += 1
            vardic[x[0]] = 'par' + str(fnum)
            t = -1
            for s in x[1].child:
                if s.name == 'type':
                    t = s.child[0].child[0].child[0].name[:-4]
                    break
            assert(t != -1)
            typedic[x[0]] = t
    return troot, vardic, typedic
def addter(root):
    if len(root.child) == 0:
        root.name += "_ter"
    for x in root.child:
        addter(x)
    return
def setProb(r, p):
    r.possibility =  p#max(min(np.random.normal(0.8, 0.1, 10)[0], 1), 0)
    for x in r.child:
        setProb(x, p)
def getLineNode(root, block, add=True):
  ans = []
  block = block + root.name
  #print(root.name, 'lll')
  for x in root.child:
    if x.name in linenode:
      if 'info' in x.getTreestr() or 'assert' in x.getTreestr() or 'logger' in x.getTreestr() or 'LOGGER' in x.getTreestr() or 'system.out' in x.getTreestr().lower():
        continue
      x.block = block
      ans.append(x)
    else:
      #print(x.name)
      s = ""
      if not add:
        s = block
        #tmp = getLineNode(x, block)
      else:
        s = block + root.name
      #print(block + root.name + "--------")
      tmp = getLineNode(x, block)
      '''if x.name == 'then_statement' and tmp == []:
        print(tmp)
        print(x.father.printTree(x.father))
        assert(0)'''
      ans.extend(tmp)
  return ans
def getroottree(tokens, isex=False):
    if isinstance(tokens[0], tuple):
        root = Node(tokens[0][0], 0)
    else:
        root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if isinstance(x, tuple):
                nnode = Node(x[0], idx)
                nnode.position = x[1]
            else:
                nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root
def ismatch(root, subroot):
    index = 0
    #assert(len(subroot.child) <= len(root.child))
    #print(len(subroot.child), len(root.child))
    for x in subroot.child:
        while index < len(root.child) and root.child[index].name != x.name:
            index += 1
        if index == len(root.child):
            return False
        if not ismatch(root.child[index], x):
            return False
        index += 1
    return True
def findSubtree(root, subroot):
    if root.name == subroot.name:
        if ismatch(root, subroot):
            return root
    for x in root.child:
        tmp = findSubtree(x, subroot)
        if tmp:
            return tmp
    return None
def generateAST(tree):
    sub = []
    if not tree:
        return ['None', '^']
    if isinstance(tree, str):
        tmpStr = tree
        tmpStr = tmpStr.replace(" ", "").replace(":", "")
        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
            tmpStr = "<string>"
        if len(tmpStr) == 0:
            tmpStr = "<empty>"
        if tmpStr[-1] == "^":
            tmpStr += "<>"
        sub.append(tmpStr)
        sub.append("^")
        return sub
    if isinstance(tree, list):
        if len(tree) == 0:
            sub.append("empty")
            sub.append("^")
        else:
            for ch in tree:
                subtree = generateAST(ch)
                sub.extend(subtree)
        return sub
    position = None
    if hasattr(tree, 'position'):
        #assert(0)
        position = tree.position
    curr = type(tree).__name__
    #print(curr)
    if True:
        if False:
            assert(0)#sub.append((str(getLiteral(tree.children)))
        else:
            sub.append((curr, position))
            try:
                for x in tree.attrs:
                    if x == "documentation":
                        continue
                    if not getattr(tree, x):
                        continue
                    '''if x == 'prefix_operators':
                        node = getattr(tree, x)
                        print(type(node))
                        print(len(node))
                        print(node[0])
                        assert(0)
                    if type(getattr(tree, x)).__name__ not in nodes:
                        print(type(getattr(tree, x)).__name__)
                        continue'''
                    sub.append(x)
                    node = getattr(tree, x)
                    if isinstance(node, list):
                        if len(node) == 0:
                            sub.append("empty")
                            sub.append("^")
                        else:
                            for ch in node:
                                subtree = generateAST(ch)
                                sub.extend(subtree)
                    elif isinstance(node, javalang.tree.Node):
                        subtree = generateAST(node)
                        sub.extend(subtree)
                    elif not node:
                        continue
                    elif isinstance(node, str):
                        tmpStr = node
                        tmpStr = tmpStr.replace(" ", "").replace(":", "")
                        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
                            tmpStr = "<string>"
                        if len(tmpStr) == 0:
                            tmpStr = "<empty>"
                        if tmpStr[-1] == "^":
                            tmpStr += "<>"
                        sub.append(tmpStr)
                        sub.append("^")
                    elif isinstance(node, set):
                        for ch in node:
                            subtree = generateAST(ch)
                            sub.extend(subtree)
                    elif isinstance(node, bool):
                        sub.append(str(node))
                        sub.append("^")
                    else:
                        print(type(node))
                        assert(0)
                    sub.append("^")
            except AttributeError:
                assert(0)
                pass
        sub.append('^')
        return sub
    else:
        print(curr)
    return sub
def getroottree2(tokens, isex=False):
    root = Node(tokens[0], 0)
    currnode = root
    idx = 1
    for x in tokens[1:]:
        if x != "^":
            nnode = Node(x, idx)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
            idx += 1
        else:
            currnode = currnode.father
    return root
'''def setProb(root, subroot, prob):
    root.possibility = max(min(max(root.possibility, prob), 0.98), 0.01)
    index = 0
    assert(len(subroot.child) <= len(root.child))
    #print(len(subroot.child), len(root.child))
    for x in subroot.child:
        while root.child[index].name != x.name:
            #print(root.child[index].name, x.name)
            index += 1
        setProb(root.child[index], x, prob)
        index += 1'''
def getSubroot(treeroot):
    currnode = treeroot
    lnode = None
    mnode = None
    while currnode:
        if currnode.name in linenode:
            lnode = currnode
            break
        currnode = currnode.father
    currnode = treeroot
    while currnode:
        if currnode.name == 'MethodDeclaration' or currnode.name == 'ConstructorDeclaration':
            mnode = currnode
            break
        currnode = currnode.father
    return lnode, mnode
def getNodeById(root, line):
    if root.position:
        if root.position.line == line and root.name != 'IfStatement' and root.name != 'ForStatement':
            return root
    for x in root.child:
        t = getNodeById(x, line)
        if t:
            return t
    return None
def containID(root):
    ans = []
    if root.position is not None:
        ans.extend([root.position.line])
    for x in root.child:
        ans.extend(containID(x))
    return ans
def getAssignMent(root):
    if root.name == 'Assignment':
        return root
    for x in root.child:
        t = getAssignMent(x)
        if t:
            return t
    return None
def isAssign(line):
    #sprint(4, line.getTreestr())
    if 'Assignment' not in line.getTreestr():
        return False
    anode = getAssignMent(line)
    if anode.child[0].child[0].name == 'MemberReference' and anode.child[1].child[0].name == 'MethodInvocation':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
            v = anode.child[1].child[0].child[0].child[0].name
        except:
            return False
        print(m, v)
        return m == v
    if anode.child[0].child[0].name == 'MemberReference':
        try:
            m = anode.child[0].child[0].child[0].child[0].name
        except:
            return False
        if "qualifier " + m in anode.child[1].getTreestr():
            return True
    return False
filepath = "code.java"
lines1 = open(filepath, "r").read().strip()
liness = lines1.splitlines()
tokens = javalang.tokenizer.tokenize(lines1)
parser = javalang.parser.Parser(tokens)
tree = parser.parse_member_declaration()
tmproot = getroottree(generateAST(tree))
lineid = eval(open("line.txt", "r").read().strip())
currroot = getNodeById(tmproot, lineid)
lnode, mnode = getSubroot(currroot)
oldcode = liness[lineid - 1]
subroot = lnode
treeroot = mnode
presubroot = None
aftersubroot = None     
linenodes = getLineNode(treeroot, "")
currid = linenodes.index(subroot)
if currid > 0:
    presubroot = linenodes[currid - 1]
if currid < len(linenodes) - 1:
    aftersubroot = linenodes[currid + 1]
setProb(treeroot, 2)
addter(treeroot)
data = []
if True:
    setProb(treeroot, 2)
    if subroot is not None:
        setProb(subroot, 1)
    if aftersubroot is not None:
        setProb(aftersubroot, 4)
    if presubroot is not None:
        setProb(presubroot, 3)
                #print(containID(subroot))
    cid = set(containID(subroot))
    maxl = -1
    minl = 1e10
    for l in cid:
        maxl = max(maxl, l - 1)
        minl = min(minl, l - 1)
                #print(maxl, liness[maxl + 1])
    precode = "\n".join(liness[0:minl])
    aftercode = "\n".join(liness[maxl + 1:])
    oldcode = "\n".join(liness[minl:maxl + 1])
    troot, vardic, typedic = solveLongTree(treeroot, subroot)
    data.append({'treeroot':treeroot, 'troot':troot, 'oldcode':oldcode, 'filepath':filepath, 'subroot':subroot, 'vardic':vardic, 'typedic':typedic, 'precode':precode, 'aftercode':aftercode, 'tree':troot.printTreeWithVar(troot, vardic), 'prob':troot.getTreeProb(troot), 'mode':0, 'line':lineid, 'isa':False})
model = test()
ans = solveone2(data, model)
tans = []
for p in ans:
    mode = p['mode']
    precode = p['precode']
    aftercode = p['aftercode']        
    oldcode = p['oldcode']
    root = getroottree2(p['code'].split())
    if '-1' in oldcode:
        continue
    if mode == 1:
        aftercode = oldcode + aftercode
    lines = aftercode.splitlines()
    if 'throw' in lines[0] and mode == 1:
        for s, l in enumerate(lines):
            if 'throw' in l or l.strip() == "}":
                precode += l + "\n"
            else:
                break
        aftercode = "\n".join(lines[s:])
    if lines[0].strip() == '}' and mode == 1:
        precode += lines[0] + "\n"
        aftercode = "\n".join(lines[1:])

    try:
        code = stringfyRoot(root, False, mode)
    except:
        print(traceback.print_exc())
        continue
    if '<string>' in code:
        if '\'.\'' in oldcode:
            code = code.replace("<string>", '"."')
        elif '\'-\'' in oldcode:
            code = code.replace("<string>", '"-"')
        elif '\"class\"' in oldcode:
            code = code.replace("<string>", '"class"')
        else:
            code = code.replace("<string>", "\"null\"")
    if len(root.child) > 0 and root.child[0].name == 'condition' and mode == 0:
        code = 'if' + code + "{"
    if code == "" and 'for' in oldcode and mode == 0:
        code = oldcode + "if(0!=1)break;"
    lnum = 0
    for l in code.splitlines():
        if l.strip() != "":
            lnum += 1
        else:
            continue
    if mode == 1 and len(precode.splitlines()) > 0 and 'case' in precode.splitlines()[-1]:
        lines = precode.splitlines()
        for i in range(len(lines) - 2, 0, -1):
            if lines[i].strip() == '}':
                break
        precode = "\n".join(lines[:i])
        aftercode = "\n".join(lines[i:]) + "\n" + aftercode
    if lnum == 1 and 'if' in code and mode == 1:
        if len(precode.splitlines()) > 0 and 'for' in precode.splitlines()[-1]:
            code = code + 'continue;\n}\n'    
        else:
            afterlines = aftercode.splitlines()
            lnum = 0
            rnum = 0
            ps = p
            for p, y in enumerate(afterlines):
                if '{' in y:
                    lnum += 1
                if '}' in y:
                    if lnum == 0:
                        aftercode = "\n".join(afterlines[:p] + ['}'] + afterlines[p:])
                            #assert(0)
                        break
                    lnum -= 1
        tmpcode = precode + "\n" + code + aftercode
        tokens = javalang.tokenizer.tokenize(tmpcode)
        parser = javalang.parser.Parser(tokens)
    else:
        tmpcode = precode + "\n" + code + aftercode
        tokens = javalang.tokenizer.tokenize(tmpcode)
        parser = javalang.parser.Parser(tokens)
    tans.append(tmpcode)
open("ans.txt", "w").write("\n\n-------\n\n".join(tans))
    