import tornado.ioloop
import tornado.web
import json
import six
import nltk
import math
import numpy as np
import collections
import unicodedata
import re
import sys
import os
from tornado.httpserver import *
#from predict import *
import threading
attr = ["NAME","ATK", "DEF", "COST","DUR","TYPE","PLAYER_CLS", "RACE","RARITY"]
dir = {"ATIS":"ATIS", "GEO":"GEO", "HS":"Compare"}
import psutil
import os
import subprocess
print ("load")
class UploadFileHandler(tornado.web.RequestHandler):
    def post(self):
        self.set_header('Access-Control-Allow-Origin', "*")
        print(self.request.body)
        content = self.request.body.decode('utf-8')
        print(content)
        data = json.loads(content)
        res = ""
        ans = "fail to generate"
        print("post")
        if(data['type'] == 'Repair'):
            open("code.java", "w").write(data['data'])
            open("line.txt", 'w').write(data['data1'])
            os.system("python3 testone.py")
        f = open("ans.txt", "r")
        ans = f.read()
        f.close()
        self.finish(ans)

app=tornado.web.Application([
    (r'/', UploadFileHandler),
])

if __name__ == '__main__':
    server = HTTPServer(app)
    server.listen(12000)
    #app.listen(12000)
    tornado.ioloop.IOLoop.instance().start()
