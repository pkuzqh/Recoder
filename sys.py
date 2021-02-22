import subprocess
p1 = subprocess.Popen("socat TCP-LISTEN:19154,fork,bind=0.0.0.0 TCP:[2001:da8:201:1088:ae1f:6bff:fe95:c902]:19925", shell=True)
p2 = subprocess.Popen("socat TCP-LISTEN:19171,fork,bind=0.0.0.0 TCP:[2001:da8:201:1088:ae1f:6bff:fe99:12be]:19925", shell=True)
p3 = subprocess.Popen("socat TCP-LISTEN:8833,fork,bind=0.0.0.0 TCP:[2001:da8:201:1088:ae1f:6bff:fe99:12be]:8833", shell=True)
p2 = subprocess.Popen("socat TCP-LISTEN:8854,fork,bind=0.0.0.0 TCP:[2001:da8:201:1088:ae1f:6bff:fe95:c902]:8833", shell=True)
p2 = subprocess.Popen("socat TCP-LISTEN:19163,fork,bind=0.0.0.0 TCP:[2001:da8:201:1088:4344:2377:633c:8142]:19925", shell=True)
p2 = subprocess.Popen("socat TCP-LISTEN:8863,fork,bind=0.0.0.0 TCP:[2001:da8:201:1088:4344:2377:633c:8142]:8833", shell=True)
import time
while(1):
    time.sleep(1)
