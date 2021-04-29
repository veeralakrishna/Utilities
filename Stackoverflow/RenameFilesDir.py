# https://stackoverflow.com/questions/66897426/renaming-files-in-the-directory-based-on-file-names-hashable-in-the-excel-sheet

"""
Error FileExistsError shows you that file with this name already exists.

So you should catch this error to continue code

    except FileExistsError: 
like
"""

import os   # PEP8: all imports at the beginnig

count = 1   # PEP8: spaces around `=`

os.chdir(r'C:\Users\Kirti\Documents\Notnumbered')

for f in os.listdir():  # if you already do `chdir` then you don't have to use full path
    try:
        file_name = '{}.pdf'.format(count)
        new_name = '{}.pdf'.format(hs[count])

        print(file_name, new_name)  # PEP8: space after `,`

        os.rename(f, new_name)
    except KeyError as ex: 
        print('KeyError:', ex)  # it is good to display message with error - to see if there was some problem
    except FileExistsError as ex: 
        print('FileExistsError:', ex)  # it is good to display message with error - to see if there was some problem

    count += 1
# Or you should check if file already exists and rename only if it doesn't exist

    if not os.path.exists(new_name):
        os.rename(f, new_name)
# like:


import os   # PEP8: all imports at the beginnig

count = 1   # PEP8: spaces around `=`

os.chdir(r'C:\Users\Kirti\Documents\Notnumbered')

for f in os.listdir():  # if you already do `chdir` then you don't have to use full path
    try:
        file_name = '{}.pdf'.format(count)
        new_name = '{}.pdf'.format(hs[count])

        print(file_name, new_name)  # PEP8: space after `,`

        if not os.path.exists(new_name):
            os.rename(f, new_name)
        else:
            print('File already exists')
    except KeyError as ex: 
        print('KeyError:', ex)  # it is good to display message with error - to see if there was some problem

    count += 1
