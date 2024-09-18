"""
Simple file that unzip all file from all folder recursively
"""

from utils.files_utils import *

# unzip all possiblity
def unzip_all(source_dir):
    for entry in os.listdir(source_dir):
        entry_path = os.path.join(source_dir, entry)
        if os.path.isdir(entry_path):
            entry_path = os.path.join(source_dir, entry)
            print('*** START UNZIP {} *** '.format(entry_path))
            unzipper(entry_path)
            print('*** END UNZIP {} *** \n'.format(entry_path))


if __name__ == '__main__':
    source_dir = '/Users/XXX/Desktop/SCAI/SCAI-SENSEI-V2'
    unzip_all(source_dir)