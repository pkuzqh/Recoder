import json
import os
import subprocess
from tqdm import tqdm
from nltk import word_tokenize
class Node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
def visitTree(node, ast):
  if isinstance(ast, dict):
      for x in ast:
          nnode = Node(x)
          nnode.father = node
          node.child.append(nnode)
          visitTree(nnode, ast[x])
  elif isinstance(ast, list):
      nnode = Node("list")
      nnode.father = node
      node.child.append(nnode)
      for x in ast:
          nn = Node("stmt")
          nnode.child.append(nn)
          nn.father = nnode
          nod = visitTree(nn, x)
          #nnode.child.append(nod)
  elif isinstance(ast, str):
      if ast == "\'\'" or ast == "\"\"":
        ast = "empty_str"
      if "\'" in ast or "\"" in ast or ast == "empty_str":
        ast = ast.replace("\"", "").replace("\'", "").replace(" ", "_").replace("%", "")
        nnode = Node("str_")
        nnode.father = node
        node.child.append(nnode)
        nnnode = Node(ast)
        nnnode.father = nnode
        nnode.child.append(nnnode)
      else:
        nnode = Node(ast)
        nnode.father = node
        node.child.append(nnode)
  elif isinstance(ast, bool):
      nnode = Node(str(ast))
      nnode.father = node
      node.child.append(nnode)
  elif not ast:
      nnode = Node("none")
      nnode.father = node
      node.child.append(nnode)
  else:
    print(type(ast))
    exit(0)
def printTree(r):
    s = r.name + " "#print(r.name)
    if len(r.child) == 0:
      s += "^ "
      return s
    r.child = sorted(r.child, key=lambda x:x.name)
    for c in r.child:
        s += printTree(c)
    s += "^ "#print(r.name + "^")
    return s
tables = json.load(open("tables.json", "r"))
tablename = {}
for t in tables:
    tablename[t['db_id']] = t 
def sovleNl(dbid, nl):
  if nl[-1] == "?":
    nl = nl[:-1] + " ?"
  elif nl[-1] == ".":
    nl = nl[:-1] + " ."
  else:
    print(nl)
  nls = nl.lower().strip().split()
  tmp = []
  for i in range(len(nls)):
    if nls[i] == "share":
      tmp.append('share_')
    if nls[i] == "females":
      tmp.append('female')
    elif "\"" in nls[i] or "\'" in nls[i] or "“" in nls[i]:
      nls[i] = nls[i].replace("\"", " | ").replace("\'", " | ").replace("“", " | ").replace("”", " | ")
      lst = nls[i].split()
      for x in lst:
        if x == "|":
          tmp.append("\"")
        elif x[-1] in ",?.!;？":
          tmp += [x[:-1].replace(",", ""), x[-1]]
        else:
          tmp.append(x)
    elif nls[i][-1] in ",?.!;？":
      tmp += [nls[i][:-1].replace(",", ""), nls[i][-1]]
    else:
      tmp.append(nls[i].replace(",", ""))
  nls = tmp
  #print(nls)
  ans = ""
  for i, x in enumerate(tablename[dbid]['table_names_original']):
    ans += x.lower() + " table_end "
    for j, y in enumerate(tablename[dbid]['column_names_original']):
      if y[0] == i:
        if y[1].lower() == "share":
          y[1] = "share_"
        ans += y[1].lower() + " " + tablename[dbid]['column_types'][j].lower() + "_end "
    ans += "col_end "
  ans += " ".join(nls)
  ans += " query_end"
  return ans
lst = json.loads(open("train_spider.json", "r").read())
for i, x in tqdm(enumerate(lst)):
  q = x['query']
  q = q.lower()
  if i == 12:
    print(q)
  open("data.txt", "w").write(q)
  status = subprocess.call(["node", "process.js"])#commands.getstatusoutput('node process.js')
  oq = ""
  if status != 0:
      oq = q
      #q = q.replace("INTERSECT", "union").replace("EXCEPT", "union")
      open("data.txt", "w").write(q)
      status = subprocess.call(["node", "process.js"])
      if status != 0:
        print(q)
        exit(1)
      #exit(1)
  s = json.load(open("data.json", "r"))
  f = open("train_output/" + str(i + 1) + ".txt", "w")
  r = Node("root")
  visitTree(r, s)
  s = printTree(r)
  f.write(s)
  f.close()
  f = open("train_input/" + str(i + 1) + ".txt", "w")
  f.write(sovleNl(x['db_id'], x['question']))
  f.close()
  