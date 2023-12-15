import os
import glob

# 현재 디렉토리 내의 모든 .py 파일 가져오기
modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))

# 각 모듈을 import
for module in modules:
    module_name = os.path.splitext(os.path.basename(module))[0]
    if module_name != "__init__":
        exec(f"from {module_name} import *")

