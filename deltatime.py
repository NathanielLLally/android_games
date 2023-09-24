from datetime import datetime
import time


then = datetime.now()
time.sleep(3.1)

now = datetime.now() - then

print(now)

