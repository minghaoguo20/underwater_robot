from datetime import datetime
import time

def c1():
    start_time = datetime.now()
    while(True):
        time.sleep(2)
        this_time = datetime.now()
        duration = this_time - start_time
        print(f"{start_time} -> {this_time} = {duration}")

def c2():
    for speed in range(10, 0, -1):
        print(speed)

if __name__ == "__main__":
    c2()


scp -r rp24@rp24.local:/home/rp24/code/underwater_robot/file/debug_output_20240611_072302 .

scp -r rp24@rp24.local:/home/rp24/code/underwater_robot/file/debug_output_20240611_101418 .
scp -r rp24@rp24.local:/home/rp24/code/underwater_robot/file/debug_output_20240611_101521 .


scp -r rp24@rp24.local:/home/rp24/code/underwater_robot/file/debug_output_20240611_130930 . 
scp -r rp24@rp24.local:/home/rp24/code/underwater_robot/file/debug_output_20240611_131030 . 
scp -r rp24@rp24.local:/home/rp24/code/underwater_robot/file/debug_output_20240611_131136 . 
