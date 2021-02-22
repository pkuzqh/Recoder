import json
import os
#lst = ['Chart-1', 'Chart-3', 'Chart-4', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-13', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-1', 'Closure-10', 'Closure-14', 'Closure-15', 'Closure-18', 'Closure-31', 'Closure-33', 'Closure-38', 'Closure-51', 'Closure-62', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-107', 'Closure-118', 'Closure-113', 'Closure-124', 'Closure-125', 'Lang-6', 'Lang-16', 'Lang-26', 'Lang-29', 'Lang-33', 'Lang-38', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Lang-61', 'Math-2', 'Math-5', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-57', 'Math-58', 'Math-59', 'Math-69', 'Math-70', 'Math-75', 'Math-80', 'Math-82', 'Math-85', 'Math-94', 'Math-105', 'Time-4', 'Time-15', 'Time-16', 'Time-19', 'Lang-43', 'Math-50', 'Time-7', 'Mockito-22', 'Closure-104', 'Math-27', 'Mockito-29', 'Mockito-38', 'Closure-33']
#lst = ['Lang-55', 'Lang-39', 'Lang-50', 'Lang-60', 'Lang-63', 'Math-88', 'Math-82', 'Math-20', 'Math-28', 'Math-6', 'Math-72', 'Math-79', 'Math-8', 'Math-98']#['Closure-38', 'Closure-123', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101', 'Math-98', 'Lang-16']
#lst = ['Closure-20', 'Closure-38', 'Closure-124', 'Lang-61', 'Math-3', 'Math-11', 'Math-48', 'Math-53', 'Math-63', 'Math-73', 'Math-101', 'Math-98', 'Lang-16']
#lst = ['Chart-1', 'Chart-8', 'Chart-9', 'Chart-11', 'Chart-12', 'Chart-20', 'Chart-24', 'Chart-26', 'Closure-14', 'Closure-15', 'Closure-62', 'Closure-63', 'Closure-73', 'Closure-86', 'Closure-92', 'Closure-93', 'Closure-104', 'Closure-118', 'Closure-124', 'Lang-6', 'Lang-26', 'Lang-33', 'Lang-38', 'Lang-43', 'Lang-45', 'Lang-51', 'Lang-55', 'Lang-57', 'Lang-59', 'Math-5', 'Math-27', 'Math-30', 'Math-33', 'Math-34', 'Math-41', 'Math-50', 'Math-57', 'Math-59', 'Math-70', 'Math-75', 'Math-80', 'Math-94', 'Math-105', 'Time-4', 'Time-7', 'Mockito-29', 'Mockito-38']
prlist = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']
ids = [range(1, 27), list(range(1, 134)), list(range(1, 66)), range(1, 107), range(1, 39), list(range(1, 28)), list(range(1, 25)), list(range(1, 23)), list(range(1, 13)), list(range(1, 15)), list(range(1, 14)), list(range(1, 40)), list(range(1, 6)), list(range(1, 64))]
#prlist = ['Cli', 'Codec', 'Collections', 'Compress', 'Csv', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath']
#ids = [list(range(1, 6)) + list(range(7, 41)), range(1, 19), range(25, 29), range(1, 48), range(1, 17), range(1, 27), range(1, 113), range(1, 7), range(1, 94), range(1, 23)]
lst = []
for k, x in enumerate(prlist):
    for y in ids[k]:
        lst.append(x + "-" + str(y))
for fs in lst:
    if not os.path.exists('patch/%s.json'%fs):
        continue
    if os.path.exists('patch/%s.txt'%fs):
        continue
    p = json.load(open('patch/%s.json'%fs, 'r'))
    wf = open("patch/%s.txt"%fs, 'w')
    for x in p:
        wf.write(str(x['mode']).strip() + "\t")
        wf.write(x['oldcode'].replace('\n', '--l--').strip() + "\t")
        wf.write(x['code'].strip() + "\n")
        #print(x['mode'], x['oldcode'], x['code'])
