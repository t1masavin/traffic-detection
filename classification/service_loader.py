import sys
from pathlib import Path
import re


def  sys_append_abs():
    file = Path('__file__').resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))



def clear_file(item):
    with open(item, 'r') as file:
        clear_file = re.sub('["\{\}]', '', file.read())

    with open(item, 'w') as file:
        file.write(clear_file)
