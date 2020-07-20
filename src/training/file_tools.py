#/usr/bin/env/python
# auxiliary tool set for file checking purposes;
# created by nebulaM78 team; capstone 2020;

# to rename a file name and obtain the classname;
def checkoff_file(src, add_vocab):
    '''
    args - 
        1. src - (string) the original path;
        2. add_vocab - (string) the word to append
    return - a list of 
        1. the modified path name
        2. the classname
    example:
        src = C:\\capstone\\a_1.txt;
        vocab = checked;
        --> return( C:\\capstone\\a_1_checked.txt, a)
    '''
    tmp = src.split('.')
    prefix = tmp[0]
    token = (prefix.split('\\'))[-1]
    classname = token.split('_')[0]
    format = tmp[-1]
    dst_checked = prefix + "_" + add_vocab + "." + format
    #print("checked\n", dst_checked)
    #print("classname\n", classname)
    return [classname, dst_checked]

# check for substring
def checksubstring(src, substring):
    '''
    args - 
        1. src - (string) the original path;
        2. substring - (string) to be checked against
    return -
        True or False   
    '''
    tmp = src.split('.')
    prefix = tmp[0]
    token = (prefix.split('\\'))[-1]
    return (substring in token)