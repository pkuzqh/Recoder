import javalang
import os
import json
def getLiteral(vals):
    for v in vals:
        if isinstance(v, str):
            print(v)
            if not(type(num(v)).__name__.strip() in nodeVect):
                global malformed
                malformed = True
            return type(num(v)).__name__
class Node:
  def __init__(self, name, s):
    self.name = name
    self.id = s
    self.father = None
    self.child = []
def visitTree(node, g):
    g.node(name=node.name + str(node.id))
    if node.father:
        g.edge(node.father.name + str(node.father.id), node.name + str(node.id))
    node.child = node.child#sorted(node.child, key=lambda x:x.name)
    for x in node.child:
        visitTree(x, g)
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
	curr = type(tree).__name__
	#print(curr)
	if True:
		if False:
			sub.append(str(getLiteral(tree.children)))
		else:
			sub.append(curr)
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
#wf = open("process.txt", "w")
if not os.path.exists('temp'):
    os.mkdir('temp')
from tqdm import tqdm
import pickle
res = []
v = 0
for i in tqdm(range(1, 1000000000)):
    if len(res) == 2000:
        open("rawdata%d.pkl" % v, "wb").write(pickle.dumps(res, protocol=4))
        v += 1
        res = []
        break
    if not os.path.exists("../szy/github_function_diff/" + str(i) + "_1.java"):
        break
    #print(i)
    lines1 = open("../szy/github_function_diff/" + str(i) + "_1.java", "r").read().strip()
    '''tokens = javalang.tokenizer.tokenize(lines1)
    parser = javalang.parser.Parser(tokens)
    try:
        #print(lines1)
        tree = parser.parse_member_declaration()
        #print(tree)
        #wf = open("temp/process%d_1.txt" % i, "w")
        treestr1 = " ".join(generateAST(tree))
        #drawtree(treestr, 'temp/' + str(i) + "_1")
       # wf.write(lines1 + "\n")
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        #print(code)
        continue'''
    lines2 = open("../szy/github_function_diff/" + str(i) + "_2.java", "r").read().strip()
    '''tokens = javalang.tokenizer.tokenize(lines2)
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
        #print(tree)
        #wf = open("temp/process%d_2.txt" % i, "w")
        treestr2 = " ".join(generateAST(tree))
        #drawtree(treestr, 'temp/' + str(i) + "_2")
        #wf.write(lines2 + "\n")
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        #print(code)
        continue
    res.append({'old':lines1, 'new':lines2, 'oldtree':treestr1, 'newtree':treestr2})'''
    res.append({'old':lines1, 'new':lines2})
open("rawdata%d.pkl" % v, "wb").write(pickle.dumps(res, protocol=4))

