#!/usr/bin/python
import commands
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import socket,SocketServer
 
PORT_NUMBER = 12000
 
class myHandler(BaseHTTPRequestHandler):
 
    #Handler for the GET requests
 
    def do_GET(self):
 
        print 'get method'
 
        return
 
try:
    #Create a web server and define the handler to manage the
    #incoming request
 
    cmd = "ip addr|grep 'scope global'|grep inet6|awk -F' ' '{print $2}'|awk -F'/' '{print $1}'"
    (status,ipv6) = commands.getstatusoutput(cmd)
 
    if ipv6 != "" :
        SocketServer.TCPServer.address_family=socket.AF_INET6
 
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
 
    #Wait forever for incoming htto requests
    server.serve_forever()
 
except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()