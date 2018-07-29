import sys
import os

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

from yolo_v3.config import Config

if __name__ == "__main__":
    cfg = Config()
    cfg.display()
    print('end')
