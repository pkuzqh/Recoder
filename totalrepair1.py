import subprocess
from tqdm import tqdm
import time
import json
import os
import signal
#lst = ['Closure-20', 'Closure-38', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101']
#lst = ['Lang-39', 'Lang-63', 'Math-88', 'Math-82', 'Math-20', 'Math-28', 'Math-6', 'Math-72', 'Math-79', 'Math-8', 'Math-98']#['Closure-38', 'Closure-123', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101', 'Math-98', 'Lang-16']
prlist = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']
ids = [range(1, 27), list(range(1, 134)), list(range(1, 66)), range(1, 107), range(1, 39), list(range(1, 28)), list(range(1, 25)), list(range(1, 23)), list(range(1, 13)), list(range(1, 15)), list(range(1, 14)), list(range(1, 40)), list(range(1, 6)), list(range(1, 64))]
'''prlist = ['Cli', 'Codec', 'Collections', 'Compress', 'Csv', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath']
ids = [list(range(1, 6)) + list(range(7, 41)), range(1, 19), range(25, 29), range(1, 48), range(1, 17), range(1, 27), range(1, 113), range(1, 7), range(1, 94), range(1, 23)]
prlist = ['JacksonDatabind', 'JacksonXml']
ids = [range(1, 113), range(1, 7)]'''
lst = []
for k, x in enumerate(prlist):
    for y in ids[k]:
        lst.append(x + "-" + str(y))
nlst = ["chart-1", "chart-3", "chart-8", "chart-9", "chart-11", "chart-20", "chart-24", "chart-26",
          "closure-5", "closure-7", "closure-14", "closure-33", "closure-57", "closure-62", "closure-63",
          "closure-73", "closure-86", "closure-92", "closure-93", "closure-104", "closure-109", "closure-115",
          "closure-118", "lang-6", "lang-26", "lang-29", "lang-33", "lang-38", "lang-39", "lang-51", "lang-55", "lang-57", "lang-59",
          "lang-63", "math-5", "math-6", "math-27", "math-30", "math-33", "math-34", "math-41", "math-50", "math-57",
          "math-65", "math-70", "math-75", "math-80", "math-94", "math-96", "math-105", "time-4", "time-7", "mockito-38"]
#lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-14', 'Closure-15', 'Closure-62', 'Closure-63', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-104', 'Closure-118', 'Closure-124', 'Lang-6', 'Lang-26', 'Lang-33', 'Lang-38', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Math-5', 'Math-27', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-50', 'Math-57', 'Math-59', 'Math-70', 'Math-75', 'Math-80', 'Math-94', 'Math-105', 'Time-4', 'Time-7', 'Mockito-29']
#lst = ['Lang-39', 'Lang-63', 'Math-6', 'Math-8', 'Math-28', 'Math-72', 'Math-79', 'Math-88']#['Math-50', 'Math-57', 'Math-59', 'Math-70', 'Math-75', 'Math-80', 'Math-94', 'Math-105', 'Time-4', 'Time-7', 'Mockito-29']
jobs = []
starttime = []
endtime = {}
for i in tqdm(range(len(lst))):
    if not os.path.exists('patchg/' + lst[i] + '.json'):
        continue
    if os.path.exists('patchesg/' + lst[i] + 'patch.txt'):
        continue
    if lst[i].lower() in nlst:
        continue
    activenum = 0
    for k, x in enumerate(jobs):
        if x.poll() is None:
            activenum += 1
            if time.time() - starttime[k] > 18000:
                os.killpg(os.getpgid(x.pid), signal.SIGTERM)#x.kill()
                activenum -= 1
        else:
            endtime[lst[k]] = time.time()
    while activenum >=5:
        activenum = 0
        for k, x in enumerate(jobs):
            if x.poll() is None:
                activenum += 1
                if time.time() - starttime[k] > 18000:
                    os.killpg(os.getpgid(x.pid), signal.SIGTERM)
                    activenum -= 1
            else:
                endtime[lst[k]] = time.time()
        time.sleep(1)
    p = subprocess.Popen('python3 repair.py %s' % lst[i], shell=True, start_new_session=True)#subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card) + " python3 testDefect4j.py " + lst[12 * i + j], stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, shell=True)#subprocess.run(["python3", "run.py"])
    jobs.append(p)
    starttime.append(time.time())
    time.sleep(10)
activenum = 0
for k, x in enumerate(jobs):
    if x.poll() is None:
        activenum += 1
        if time.time() - starttime[k] > 18000:
            os.killpg(os.getpgid(x.pid), signal.SIGTERM)#x.kill()
            activenum -= 1
    else:
        endtime[lst[k]] = time.time()
while activenum != 0:
    activenum = 0
    for k, x in enumerate(jobs):
        if x.poll() is None:
            activenum += 1
            if time.time() - starttime[k] > 18000:
                os.killpg(os.getpgid(x.pid), signal.SIGTERM)
                activenum -= 1
        else:
            endtime[lst[k]] = time.time()
    time.sleep(1)
open('timep.json', 'w').write(json.dumps(endtime))
