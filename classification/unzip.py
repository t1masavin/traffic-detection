import patoolib
import glob
ROOT = '/root/workspace/work/Digital-Tashkent/Signs/data'
file_list = glob.glob(ROOT + '/rar_file/*')
for file in file_list:
    patoolib.extract_archive(file, outdir=ROOT + '/content')