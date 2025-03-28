import json
import os
from utils.iojson import *

def main():
    file_path = "/home/rp24/code/underwater_robot/flag.json"
    dic = {"final_flag": False}
    print(dic)
    json.dump(dic, open(file_path, "w+"))

    result_save_dir = "/home/rp24/code/underwater_robot/result"
    if os.path.exists(result_save_dir):
        files = os.listdir(result_save_dir)
        for file in files:
            os.remove(os.path.join(result_save_dir, file))

if __name__ == "__main__":
    main()
