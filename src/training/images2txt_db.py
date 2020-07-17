#/usr/bin/env/python
# to create a database of all the reformatted images in txt;
# created by nebulaM78 team; capstone 2020;


# structure;
#   1. convert all the images to json
#   2. json to txt format
#   3. compile all the txt files;
#   * while maintaining the right label;


import images2json as i2j
import json_image2txt as ji2t
import tempfile # needed for temporary files/directory;

	
# assume this image format to be fixed throughout; 
# shall handle other formats eventually
imageformat = "png"
	
def images2txt_db(imagfolder_path, db_path, imageformat):
    '''
    args:
        1. imagfolder_path; the folder with all the raw images; e.g.
            #   folder: alphabet-1
            #       images: 
            #          -> A1.png
            #           ...
            #          -> Z1.png
        2. db_path; save all the converted image_txt to ...?
    return:
        none;
    function:
        1. # to create a database of all the reformatted images in txt;
    '''
    
    # create a directory to store all the json files;
    # once all images have been processed; 
    # convert these json to txt and save it to the final destionation
    dummydirec = tempfile.TemporaryDirectory()
    dummy_path = dummydirec.name
    image2json(src_path, dummy_path, imageformat)





    # reminder; once all done; delete the temp folder for good practice;
    # note: once this script is done running; the temp direc will be deleted regardless;
    dummydirec.cleanup()

    # test driver;
if __name__ == '__main__':
    N = 5
    # fixed location for all converted images;
        write_path = "C:\\Users\\yongw4\\Desktop\\image-database\\"
    
    for i in range(1, N):
        tmp = "C:\\Users\\yongw4\\Desktop\\alphabets\\alphabets-{CHANGE}\\"
        imageformat = "png"
        src_path = tmp.format(CHANGE = i)
        print(src_path)
        
        