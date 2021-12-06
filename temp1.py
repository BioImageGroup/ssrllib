import time
import sys
import os

if __name__ == "__main__":
    try:
        wait_time = int(sys.argv[1])
    except:
        wait_time = 2

    print(f'im gonna wait {wait_time} secs')
    time.sleep(wait_time)
    print(3)
