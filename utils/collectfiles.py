from zipfile import ZipFile
import os
from os.path import basename


def get_all_file_paths(directory):
  
    # initializing empty file paths list
    file_paths = []
  
    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
  
    # returning all file paths
    return file_paths


def zip_files(sourcedir=os.getcwd(),targetdir=os.getcwd(),target_file_name='code.zip',create_subfolders=True,allowed_extn=[],blacklisted_extn=[]):
    file_paths = get_all_file_paths(sourcedir)
    with ZipFile(os.path.join(targetdir,target_file_name), 'w') as zipObj:
       # Iterate over all the files in directory
       for file in file_paths:
        if(basename(file).split('.')[-1] in allowed_extn or len(allowed_extn)==0 or allowed_extn is None ):
            if(not (basename(file).split('.')[-1] in blacklisted_extn)):
                   # Add file to zip
                if create_subfolders:
                    zipObj.write(file, file)
                else:
                    zipObj.write(file, basename(file))
