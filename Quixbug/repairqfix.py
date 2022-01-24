import json
import sys
import os
from Searchnode import Node
from stringfycode import stringfyRoot
import javalang
import subprocess
import time
import signal
import traceback
#lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-13', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-1', 'Closure-10', 'Closure-14', 'Closure-15', 'Closure-18', 'Closure-31', 'Closure-33', 'Closure-38', 'Closure-51', 'Closure-62', 'Closure-63', 'Closure-70', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-107', 'Closure-118', 'Closure-113', 'Closure-124', 'Closure-125', 'Closure-129', 'Lang-6', 'Lang-16', 'Lang-26', 'Lang-29', 'Lang-33', 'Lang-38', 'Lang-39', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Lang-61', 'Math-2', 'Math-5', 'Math-25', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-57', 'Math-58', 'Math-59', 'Math-69', 'Math-70', 'Math-75', 'Math-80', 'Math-82', 'Math-85', 'Math-94', 'Math-105', 'Time-4', 'Time-15', 'Time-16', 'Time-19', 'Lang-43', 'Math-50', 'Math-98', 'Time-7', 'Mockito-38', 'Mockito-22', 'Mockito-29', 'Mockito-34', 'Closure-104', 'Math-27']
lst = ['Lang-39', 'Lang-63', 'Math-88', 'Math-82', 'Math-20', 'Math-28', 'Math-6', 'Math-72', 'Math-79', 'Math-8', 'Math-98']#['Closure-38', 'Closure-123', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101', 'Math-98', 'Lang-16']
bugid = sys.argv[1]
def convert_time_to_str(time):
    #时间数字转化成字符串，不够10的前面补个0
    if (time < 10):
        time = '0' + str(time)
    else:
        time=str(time)
    return time

def sec_to_data(y):
    h=int(y//3600 % 24)
    d = int(y // 86400)
    m =int((y % 3600) // 60)
    s = round(y % 60,2)
    h=convert_time_to_str(h)
    m=convert_time_to_str(m)
    s=convert_time_to_str(s)
    d=convert_time_to_str(d)
    # 天 小时 分钟 秒
    return d + ":" + h + ":" + m + ":" + s
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
import xml.etree.ElementTree as ET

def get_maven_test_results(bug, working_directory):
    errors = 0
    failures = 0
    tests = 0
    skips = 0
    for rootPath, dirs, files in os.walk(working_directory, topdown=False):
        if "surefire-reports" not in rootPath:
            continue
        for name in files:
            if ".xml" not in name:
                    continue
            try:
                tree = ET.parse(os.path.join(rootPath, name))
                root = tree.getroot()
                if 'errors' in root.attrib:
                    errors += int(root.attrib['errors'])
                if 'failures' in root.attrib:
                    failures += int(root.attrib['failures'])
                if 'failed' in root.attrib:
                    failures += int(root.attrib['failed'])
                if 'tests' in root.attrib:
                    tests += int(root.attrib['tests'])
                if 'skipped' in root.attrib:
                    skips += int(root.attrib['skipped'])
            except:
                continue
    return {'tests': tests, 'failures': failures, 'errors': errors, 'skips': skips}
lst = [bugid]
starttime = time.time()
timelst = []
for x in lst:
    wf = open('patches/' + x + "patch.txt", 'w')
    patches = json.load(open("patch/%s.json"%x, 'r'))
    curride = ""
    #x = x.replace("-", "")
    x = x[:-5]
    if os.path.exists('buggy%s' % x):
        os.system('rm -rf buggy%s' % x)
    os.system("cp -r ../qbugs/%s buggy%s"%(x, x) )
    xsss = x
    for i, p in enumerate(patches):
        if i < 0:
            continue
        endtime = time.time()
        if endtime - starttime > 18000:
            open('timeg.txt', 'a').write(xsss + "\t" + sec_to_data(endtime - starttime) + "\n")
            exit(0)
        try:
            root = getroottree2(p['code'].split())
        except:
            #assert(0)
            continue
        mode = p['mode']
        precode = p['precode']
        aftercode = p['aftercode']        
        oldcode = p['oldcode']
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
        #print(aftercode.splitlines()[:10])

        try:
            code = stringfyRoot(root, False, mode)
        except:
            #print(traceback.print_exc())
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
            if 'while' in oldcode:
                code = 'while' + code + "{"
            elif '} else if' in oldcode:
                code = '}else if' + code + "{"
            elif 'else if' in oldcode:
                code = 'else if' + code + "{"
            
            else:
                code = 'if' + code + "{"
        if code == "" and 'for' in oldcode and mode == 0:
            code = oldcode + "if(0!=1)break;"
        filepath2 = 'buggy%s/src/main/java/%s.java'%(x, x)
        lnum = 0
        for l in code.splitlines():
            if l.strip() != "":
                lnum += 1
            else:
                continue
        if 'ArrayList<Integer>' in oldcode:
            code = code.replace('ArrayList', 'ArrayList<Integer>')
        if mode == 1 and len(precode.splitlines()) > 0 and 'case' in precode.splitlines()[-1]:
            lines = precode.splitlines()
            for i in range(len(lines) - 2, 0, -1):
                if lines[i].strip() == '}':
                    break
            precode = "\n".join(lines[:i])
            aftercode = "\n".join(lines[i:]) + "\n" + aftercode
        if lnum == 1 and 'if' in code and mode == 1:
            if p['isa']:
                code = code.replace("if", 'while')
            #print('ppp', precode.splitlines()[-1])
            if len(precode.splitlines()) > 0 and 'for' in precode.splitlines()[-1]:
                code = code + 'continue;\n}\n'    
            else:
                afterlines = aftercode.splitlines()
                lnum = 0
                rnum = 0
                ps = p
                for p, y in enumerate(afterlines):
                    if ps['isa'] and y.strip() != '':
                        aftercode = "\n".join(afterlines[:p + 1] + ['}'] + afterlines[p + 1:])
                        break
                    if '{' in y:
                        lnum += 1
                    if '}' in y:
                        if lnum == 0:
                            aftercode = "\n".join(afterlines[:p] + ['}'] + afterlines[p:])
                            #assert(0)
                            break
                        lnum -= 1
            print(code)
            tmpcode = precode + "\n" + code + aftercode
            tokens = javalang.tokenizer.tokenize(tmpcode)
            parser = javalang.parser.Parser(tokens)
        else:
            print(code)
            tmpcode = precode + "\n" + code + aftercode
            tokens = javalang.tokenizer.tokenize(tmpcode)
            parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse()
        except:
            #assert(0)
            #print(code)
            #assert(0)
            #print('ttttt')
            continue
        print(filepath2)
        open(filepath2, "w").write(tmpcode)

        bugg = False
        if not bugg:
            print('s')
            cmd = 'cd buggy%s && mvn clean && mvn test' % x
            Returncode = ""
            child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, start_new_session=True)
            while_begin = time.time() 
            while True:                
                Flag = child.poll()
                #print(time.time() - while_begin, Flag)
                if  Flag == 0:
                    Returncode = child.stdout.readlines()#child.stdout.read()
                    break
                elif Flag != 0 and Flag is not None:
                    bugg = True
                    break
                elif time.time() - while_begin > 180:
                    os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    bugg = True
                    break
                else:
                    time.sleep(1)
            log = get_maven_test_results(x, 'buggy%s'%x)
            print('log', log)
            #print(Returncode)
            if log['errors'] + log['failures'] == 0 and log['tests'] != 0:
                print('success')
                endtime = time.time()
                open('timeg.txt', 'a').write(xsss + "\t" + sec_to_data(endtime - starttime) + "\n")
                #timelst.append(sec_to_data(endtime - starttime))
                wf.write(curride + "\n")
                wf.write("-" + oldcode + "\n")
                wf.write("+" +  code + "\n")
                wf.write("🚀\n")
                wf.flush()    
                if os.path.exists('buggy%s' % x):
                    os.system('rm -rf buggy%s' % x)
                exit(0)
        #exit(0)

    endtime = time.time()

