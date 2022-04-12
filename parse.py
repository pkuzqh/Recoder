import javalang
import javalang.tree
import pathlib
import os
import shutil
import subprocess
import json

def get_name(obj):
    if obj is None:
        return None
    else:
        return obj.name

def parse_file(p):
    with p.open(encoding='utf-8',errors='ignore') as f:
        src_code=f.read()
        try:
            tree=javalang.parse.parse(src_code)
        except javalang.parser.JavaSyntaxError as e:
            print('!! syntax error in file',p)
            return None

    pkg_name=tree.package.name
    clses=[]
    for sub in tree.types:
        if isinstance(sub,(javalang.tree.ClassDeclaration,javalang.tree.InterfaceDeclaration,javalang.tree.EnumDeclaration)):
            methods=[]
            fields=[]

            for method in sub.methods:
                params=[]
                for param in method.parameters:
                    params.append({
                        'type': get_name(param.type),
                        'name': param.name,
                    })

                methods.append({
                    'type': get_name(method.return_type),
                    'name': method.name,
                    'params': params,
                })
            for field in sub.fields:
                vartype=field.type.name
                for var in field.declarators:
                    fields.append({
                        'type': vartype,
                        'name': var.name,
                    })

            clses.append({
                'name': sub.name,
                'methods': methods,
                'fields': fields,
            })
        elif isinstance(sub,(javalang.tree.AnnotationDeclaration,)):
            pass # ignore these types
        else:
            print('!! unknown',type(sub),'in',p)

    return {
        'package_name': pkg_name,
        'filename': p.name,
        'classes': clses,
    }

def parse_package(pkg_name,ver_tag):
    pathlib.Path('var').mkdir(exist_ok=True)

    print('== package %s ver %s'%(pkg_name,ver_tag))

    # checkout

    print(' checking out')

    retcode=os.system('defects4j checkout -p %s -v %sb -w var/code_root 1>/dev/null 2>&1'%(pkg_name,ver_tag))
    assert retcode==0

    # get class root

    retcode=os.system('defects4j export -p dir.src.classes -w var/code_root -o var/class_root_path.txt 1>/dev/null 2>&1')
    assert retcode==0
    with open('var/class_root_path.txt') as f:
        cls_root=f.read()

    # run

    print(' running')

    ret=[]

    for p in (pathlib.Path('var/code_root')/cls_root).glob('**/*.java'):
        r=parse_file(p)
        if r is not None:
            ret.append(r)

    # save and cleanup

    print(' saving and cleaning up')

    pathlib.Path('result').mkdir(exist_ok=True)
    with pathlib.Path('result/%s-%s.json'%(pkg_name,ver_tag)).open('w',encoding='utf-8') as f:
        json.dump(ret,f,indent=1)

    shutil.rmtree('var')

#for p in (pathlib.Path('var/code_root')).glob('**/*.java'):
#    parse_file(p)

def enum_package_and_ver():
    for proj in subprocess.check_output(['defects4j','pids']).splitlines():
        proj=proj.decode()
        if proj[0] in ('C','G','J'):
            continue
        for ver in subprocess.check_output(['defects4j','bids','-p',proj]).splitlines():
            ver=int(ver.decode())
            yield proj,ver

for pr,ver in enum_package_and_ver():
    parse_package(pr,ver)