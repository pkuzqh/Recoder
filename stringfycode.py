import pickle
import traceback
def stringfyNode(root):
    #print(root.getTreestr())
    if root.name == 'LocalVariableDeclaration' or root.name == 'VariableDeclaration':
        typestr = ""
        idenname = ""
        mod = ""
        for x in root.child:
            if x.name == 'type':
                typestr = stringfyNode(x)
            if x.name == 'modifiers':
                mod = stringfyNode(x)
            if x.name == 'declarators':
                idenname = stringfyNode(x)
        return mod + typestr + " " + idenname + ";\n"
    elif root.name == 'modifiers':
        ans = ""
        for x in root.child:
            ans += x.name[:-4] + " "
        return ans
    elif root.name == 'BlockStatement' or root.name == 'parameter' or root.name == 'return_type' or root.name == 'type' or root.name == "initializer" or root.name == 'expression':
        #print(root.getTreestr())
        assert(len(root.child) == 1)
        return stringfyNode(root.child[0])
    elif root.name == 'BreakStatement_ter':
        return 'break;\n'
    elif root.name == 'BasicType':
        #print(root.getTreestr())
        t = ""
        dim = ""
        for x in root.child:
            if x.name == 'name':
                t = stringfyNode(x)
            elif x.name == 'dimensions':
                dim = stringfyNode(x)
            else:
                print(x.name)
                assert(0)
        return t + dim
    elif root.name == "None_ter":
        return ""
    elif 'Exception_ter' in root.name:
        return root.name[:-4]
    elif root.name == 'DoStatement':
        condition = ""
        body = ''
        for x in root.child:
            if x.name == 'condition':
                condition = stringfyNode(x)
            if x.name == 'body':
                body = stringfyNode(x)
        return 'do{\n' + body + "}while" + condition + "\n"
    elif root.name == "dimensions":
        ans = ""
        for x in root.child:
            if x.name == "None_ter":
                ans = "[]"
            else:
                ans = stringfyNode(x)
        return ans
    elif root.name == 'SwitchStatement':
        return ''
    elif root.name == 'AssertStatement':
        return ''
    elif root.name == 'ContinueStatement_ter':
        return 'continue;\n'
    elif root.name == 'ContinueStatement':
        return 'continue ' + root.child[0].child[0].name + ';\n'
    elif root.name == 'BlockStatement_ter':
        return ''
    elif root.name == 'MethodReference':
        exp = ""
        method = ""
        args = "("
        for x in root.child:
            if x.name == 'expression':
                exp = stringfyNode(x)
            if x.name == 'method':
                method = stringfyNode(x.child[0])
            if x.name == 'type_arguments':
                args += stringfyNode(x)
        args += ")"
        ans = exp + "." + method + args
        return ans
    elif root.name == 'WhileStatement':
        con = ""
        body = ""
        for x in root.child:
            if x.name == 'condition':
                con = 'while' + stringfyNode(x)
            if x.name == 'body':
                body = stringfyNode(x)
            #if x.name == 'else_statement':
            #    elses = stringfyNode(x.child[0])
        return con + '{\n' + body + "\n}"
    elif root.name == 'ArrayInitializer':
        ans = '{'
        for x in root.child:
            if x.name == 'initializers':
                for y in x.child:
                    ans += stringfyNode(y) + ","
        return ans[:-1] + '}'
    elif root.name == 'TypeArgument':
        t = ""
        t2 = ""
        #print(root.getTreestr())
        for x in root.child:
            if x.name == 'type':
                t = stringfyNode(x)
            elif x.name == 'pattern_type':
                t = x.child[0].name[:-4]
            else:
                t2 = x.child[0].name[:-4]
        #print(t, "ppppp")
        return t + t2
    elif root.name == 'LambdaExpression':
        return ""
    elif root.name == 'ArrayCreator':
        #print(root.getTreestr())
        t = ""
        dim = '[]'
        ini = ""
        for x in root.child:
            if x.name == 'type':
                t = stringfyNode(x)
            if x.name == 'dimensions':
                dim = stringfyNode(x)
            if x.name == 'initializer':
                ini = stringfyNode(x)
        return 'new ' + t + "[" + dim + "]" + ini
    elif root.name == 'IfStatement':
        con = ""
        then = ""
        elses = ""
        for x in root.child:
            if x.name == 'condition':
                con = 'if' + stringfyNode(x)
            if x.name == 'then_statement':
                then = stringfyNode(x.child[0])
            if x.name == 'else_statement':
                elses = stringfyNode(x.child[0])
        if then == "":
            return con + "{\n"
        if elses == "":
            return con + '{\n' + then + "\n}"
        return con + '{\n' + then + "\n}" + 'else{\n' + elses + "\n}"
    elif root.name == 'ReferenceType':
        args = ""
        n = ""
        sub = ""
        dim = ""
        for x in root.child:
            if x.name == 'arguments':
                args = stringfyNode(x)
            if x.name == 'name':
                n = stringfyNode(x)
            if x.name == 'subtype':
                sub = stringfyNode(x.child[0])
            if x.name == 'dimensions':
                dim = stringfyNode(x)
        print('arg', args)
        if dim != "":
            n = n + " " + dim
        if args != "":
            ans = n +"<" + args + ">"
        else:
            ans = n
        if sub != "":
            ans = ans + "." + sub
            return ans
        else:
            return ans

    elif root.name == 'StatementExpression':
        return stringfyNode(root.child[0]) + ";\n"
    elif root.name == 'body' or root.name == 'catches' or root.name == 'statements' or root.name == 'finally_block':
        ans = ""
        for x in root.child:
            ans += stringfyNode(x)
        return ans
    elif root.name == 'TernaryExpression':
        con = ""
        iftrue = ""
        iffalse = ""
        for x in root.child:
            if x.name == 'condition':
                con = stringfyNode(x)
            if x.name == 'if_true':
                iftrue = stringfyNode(x.child[0])
            if x.name == 'if_false':
                iffalse = stringfyNode(x.child[0])
        return con + "?" + iftrue + ":" + iffalse 
    elif root.name == 'ForStatement':
        con = ""
        body = ""
        for x in root.child:
            if x.name == 'control':
                con = stringfyNode(x.child[0])
            if x.name == 'body':
                body = stringfyNode(x)
        #print(root.getTreestr())
        return 'for' + con + "{\n" + body + "\n}"
    elif root.name == 'control':
        return stringfyNode(root.child[0])
    elif root.name == 'ForControl':
        ini = ""
        con = ""
        up = ""
        for x in root.child:
            if x.name == 'init':
                ini = stringfyNode(x.child[0]).strip()
            if x.name == 'condition':
                con = stringfyNode(x)[1:-1]
            if x.name == "update":
                up = stringfyNode(x.child[0])
        if up == '':
            return '(' + con + ")"
        return 'for(' + ini + con + ";" + up + ") {"
    elif root.name == 'ForControl_ter':
        return '(;;)'
    elif root.name == 'EnhancedForControl':
        var = ""
        itera = ""
        for x in root.child:
            if x.name == 'var':
                var = stringfyNode(x.child[0])
            if x.name == 'iterable':
                itera = stringfyNode(x.child[0])
        return '(' + var + ":" + itera + ")"
    elif root.name == 'CatchClause':
        p = ""
        b = ""
        for x in root.child:
            if x.name == 'parameter':
                p = stringfyNode(x)
            if x.name == 'block':
                b = stringfyNode(x)
        return "catch(" + p + "){\n" + b + "\n}" 
    elif root.name == 'MethodDeclaration':
        rettype = ""
        methodname = ""
        throws = ""
        args = ""
        body = ""
        for x in root.child:
            if x.name == 'return_type':
                rettype = stringfyNode(x)
            if x.name == 'name':
                methodname = stringfyNode(x)
            if x.name == 'throws':
                throws = "throws " + stringfyNode(x) 
            if x.name == 'parameters':
                for y in x.child:
                    args += stringfyNode(y) + ","
            if x.name == 'body':
                body = stringfyNode(x)
        if rettype == "":
            rettype = 'void'
        return rettype + " " + methodname + " (" + args[:-1] + ") " + throws + " {\n" + body + " }" 
    elif root.name == 'parameters':
        args = ""
        for x in root.child:
            args += stringfyNode(x) + ","
        return args[:-1]
    elif root.name == 'ExplicitConstructorInvocation':
        methodname = "this("
        for x in root.child:
            if x.name == 'arguments':
                methodname += stringfyNode(x)
            else:
                assert(0)
        methodname += ");"
        return methodname
    elif root.name == 'Assignment':
        ass = ""
        val = ""
        type = ""
        for x in root.child:
            if x.name == 'expressionl':
                ass = stringfyNode(x.child[0])
            if x.name == 'value':
                val = stringfyNode(x.child[0])
            if x.name == 'type':
                type = x.child[0].name[:-4]
        return ass + " %s "%type + val
    elif root.name == 'selectors':
        ans = ""
        for x in root.child:
            ans += stringfyNode(x) + "."
        return ans[:-1]
    elif root.name == 'MemberReference':
        qual = ""
        member = ""
        post = ""
        pre = ""
        sel = ""
        for x in root.child:
            if x.name == 'qualifier':
                qual = stringfyNode(x)#x.child[0].name[:-4] + "."
            if x.name == "member":
                member = stringfyNode(x)
            if x.name == "postfix_operators":
                post = stringfyNode(x)
            if x.name == 'prefix_operators':
                pre = stringfyNode(x)
            if x.name == 'selectors':
                sel = stringfyNode(x)
        return pre + qual + member + sel + post
    elif root.name == 'name' or root.name == 'member':
        return root.child[0].name[:-4]
    elif root.name == 'declarators':
        ans = ""
        for x in root.child:
            ans += stringfyNode(root.child[0]) 
            ans += ","
        return ans[:-1]
    elif root.name == 'VariableDeclarator':
        #print(root.getTreestr())
        idenname = ""
        clname = ""
        for x in root.child:
            if x.name == 'name':
                idenname = stringfyNode(x)
            if x.name == 'initializer':
                clname = stringfyNode(x)
        #print(idenname)
        #print(clname)
        return idenname + " = " + clname
    elif root.name == 'ArraySelector':
        qual = ""
        index = ""
        post = ""
        pre = ""
        sel = ""
        for x in root.child:
            if x.name == 'qualifier':
                qual = stringfyNode(x)#x.child[0].name[:-4] + "."
            if x.name == "index":
                index = stringfyNode(x.child[0])
            if x.name == "postfix_operators":
                post = stringfyNode(x)
            if x.name == 'prefix_operators':
                pre = stringfyNode(x)
            if x.name == 'selectors':
                sel = stringfyNode(x)
        return "[" + pre + qual + index + sel + post + "]"
    elif root.name == 'prefix_operators' or root.name == "postfix_operators":
        return root.child[0].name[:-4]
    elif root.name == 'ClassCreator':
        arguments = '('
        for x in root.child:
            if x.name == 'type':
                typename = stringfyNode(x)
            if x.name == 'arguments':
                arguments += stringfyNode(x)
        arguments += ")"
        return 'new ' + typename + arguments
    elif root.name == 'arguments':
        argstr = ""
        for i, x in enumerate(root.child):
            if i == 0:
                argstr += stringfyNode(x)
            else:
                argstr += ", " + stringfyNode(x)
        return argstr
    elif root.name == 'SuperConstructorInvocation':
        methodname = "super("
        for x in root.child:
            if x.name == 'arguments':
                methodname += stringfyNode(x)
            else:
                assert(0)
        methodname += ")"
        return methodname
    elif root.name == 'condition':
        return '(' + stringfyNode(root.child[0]) + ')'
    elif root.name == 'BinaryOperation':
        operatorname = ""
        operatorleft = ""
        operatorright = ""
        for x in root.child:
            if x.name == 'operator':
                operatorname = x.child[0].name[:-4]
            if x.name == 'operandl':
                operatorleft = stringfyNode(x.child[0])
            if x.name == 'operandr':
                operatorright = stringfyNode(x.child[0])
        return "(" + operatorleft + " " + operatorname + " " + operatorright + ")"
    elif root.name == 'Literal':
        pre = ''
        post = ''
        v = ''
        for x in root.child:
            if x.name == 'prefix_operators':
                pre = stringfyNode(x)
            if x.name == 'postfix_operators':
                post = stringfyNode(x)
            if x.name == 'value':
                v = x.child[0].name[:-4]
        ans = pre + v + post
        return ans
    elif root.name == 'ConstructorDeclaration':
        name = ""
        args = ""
        body = ""
        throws = ""
        for x in root.child:
            if x.name == 'name':
                name = stringfyNode(x)
            if x.name == 'parameters':
                args += stringfyNode(y) + ","
            if x.name == 'throws':
                throws = 'throws ' + stringfyNode(x)
            if x.name == 'body':
                body = stringfyNode(x)
        return 'class ' + name + " " + throws + "{\n" + body + "}\n"
    elif root.name == 'SynchronizedStatement':
        lock = ""
        block = ""
        for x in root.child:
            if x.name == 'lock':
                lock = stringfyNode(x.child[0])
            if x.name == "block":
                block = stringfyNode(x)
        return "synchronized(" + lock + "){\n" + block + "}" 
    elif root.name == 'throws':
        thows = ""
        for x in root.child:
            thows += stringfyNode(x) + " "
        return thows.strip()
    elif root.name == 'This':
        ansname = 'this.'
        member = ""
        qual = ""
        for x in root.child:
            if x.name == 'selectors':
                member = stringfyNode(x)
            if x.name == 'qualifier':
                qual = stringfyNode(x)
        if qual == "":
            ansname += member
        else:
            ansname += qual + "." + member
        return ansname
    elif root.name == 'ReturnStatement_ter':
        return 'return;\n'
    elif root.name == 'qualifier':
        if len(root.child[0].child) == 0:
            return root.child[0].name[:-4] + "."
        return stringfyNode(root.child[0]) + "."
        #return root.child[0].name[:-4] + "."
    elif root.name == 'MethodInvocation' or root.name == 'SuperMethodInvocation':
        ans = ""
        member = ""
        arguments = "("
        selectors = ""
        qual = ""
        pre = ""
        for x in root.child:
            if x.name == 'member':
                member += stringfyNode(x)
            if x.name == 'arguments':
                arguments += stringfyNode(x)
            if x.name == 'qualifier':
                qual = stringfyNode(x)
            if x.name == 'selectors':
                selectors = "." + stringfyNode(x)
            if x.name == 'prefix_operators':
                pre = stringfyNode(x)
        arguments = arguments + ")"
        ans = pre + qual + member + arguments + selectors
        if root.name == 'SuperMethodInvocation':
            ans = 'super.' + ans
        return ans
    elif root.name == 'FormalParameter' or root.name == 'CatchClauseParameter':
        t = ""
        na = ""
        for x in root.child:
            if x.name == 'type':
                t = stringfyNode(x)
            if x.name == 'name':
                na = stringfyNode(x)
        return t + " " + na
    elif root.name == 'TryStatement':
        block = ""
        ca = ""
        fin = ""
        for x in root.child:
            if x.name == 'block':
                block = stringfyNode(x)
            if x.name == 'catches':
                ca = stringfyNode(x)
            if x.name == 'finally_block':
                fin = stringfyNode(x)
        return 'try{\n' +  block + '}\n' + ca + 'finally{\n' + fin + "}\n"
    elif root.name == 'block':
        ans= ""
        for x in root.child:
            ans += stringfyNode(x) + "\n"
        return ans


    elif root.name == 'throws_ter':
        return ""
    elif root.name == 'annotations':
        return ""
    elif root.name == 'ReturnStatement':
        ans = "return " + stringfyNode(root.child[0]) + ";"
        return ans
    elif root.name == 'ReturnStatement_er':
        ans = "return;"
        return ans
    elif root.name == 'ThrowStatement':
        return ""
    elif root.name == 'This_ter':
        return 'this'
    elif root.name == 'modifiers' or root.name == 'modifiers_ter':
        return ""
    elif root.name == 'assertEquals' or root.name == 'assertEquals_ter':
        return ""
    elif root.name == 'ClassReference':
        print(root.getTreestr())
        #assert(len(root.child) == 1)
        t = ""
        s = ""
        for x in root.child:
            if x.name == 'type':
                t = stringfyNode(x)
            if x.name == 'selectors':
                s = stringfyNode(x)
        return t + s
    elif root.name == 'Cast':
        types = ""
        exp = ""
        for x in root.child:
            if x.name == 'type':
                types = stringfyNode(x)
            if x.name == 'expression':
                exp = stringfyNode(x)
        return "(" + types + ")" + exp
    else:
        print(1, root.name)
        assert(0)
    #elif root.name == 'MemberReference':

def stringfyRoot(root, isIf, mode):
    p = ""
    for x in root.child:
        print(x.getTreestr())
        s = stringfyNode(x)
        try:
            s = stringfyNode(x)
        except:
            #print(traceback.print_exc())
            s = ""
        if x.name == 'IfStatement':
            if isIf == 1:
                if s == "":
                    s = "if(True){\n"
        p += s
    if len(root.child) > 0 and root.child[0].name == 'IfStatement' and len(root.child) > 1 and mode == 1:
        p += "}"   
    return p
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
        self.treestr = ""
        self.hasmatch = False
        self.possibility = 0.01
    def getprob(self):
        ans = [self.possibility]
        for x in self.child:
            ans.extend(x.getprob())
        return ans
    def printTree(self, r):
        #s = r.name + " "#print(r.name)
        if len(r.child) == 0:
            s = r.name + '_ter' + " "
            s += "^ "
            return s
        s = r.name + " "
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "#print(r.name + "^")
        return s
    def getTreestr(self):
        if self.treestr == "":
            self.treestr = self.printTree(self)
            return self.treestr
        else:
            return self.treestr
        return self.treestr
def getroottree(tokens, isex=False):
    if isex:
        root = Node(tokens[0], 0)
    else:
        root = Node('MethodDeclaration', 0)
    currnode = root
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if tokens[i + 2] == '^' and '_ter' not in x:
                x += '_ter'
            nnode = Node(x, i + 1)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
        else:
            currnode = currnode.father
    return root
'''data = open('outval1.txt', "r").readlines()#pickle.load(open('process_datacopy.pkl', 'rb'))
for i, x in enumerate(data):
    r = getroottree(x.split())
    print(i)
    print(x)
    try:
        s = stringfyNode(r.child[0])
        print(s)
    except:
        print(x)'''
    #assert(0)
#root = getroottree('root LocalVariableDeclaration type ReferenceType name Double_ter ^ ^ ^ ^ declarators VariableDeclarator name objTypePair_ter ^ ^ initializer ClassCreator type ReferenceType name Double_ter ^ ^ ^ ^ arguments MemberReference member x_ter ^ ^ ^ ^ ^ ^ ^ ^ ^ StatementExpression expression SuperConstructorInvocation arguments MemberReference member f_ter ^ ^ ^ ^ ^ ^ ^ ^ '.split())
#print(stringfyRoot(root))