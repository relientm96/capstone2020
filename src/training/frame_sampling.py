#/usr/bin/env/python
# tools to (down) resample the frame in the form of txt for training;

# idea inspired by matthew;
# created by nebulaM78 team; capstone 2020;

import numpy as np
import random
import LSTM_tools as lstm
TEST_PATH = "C:\\Users\\yongw4\\Desktop\\X_train.txt"
# global constants;
FRAMES = 75
CUT_FRAMES = 70
FEATURE = 98
HALF = CUT_FRAMES/2

# remove all the empty frames;
'''
X_path = TEST_PATH
n_steps = 75
file = open(X_path, 'r')
X_ = np.array(
	[elem for elem in [
		row.split(',') for row in file
	]], 
	dtype=np.float32
)
file.close()
blocks = int(len(X_) / n_steps)
X_ = np.array(np.split(X_,blocks))

print(X_.shape)
print(X_)
'''
'''
arr = np.array([[[1,1,1],[0,0,0]],[[2,2,2], [0,0,0]],[[3,3,3], [0,0,0]]])
arr = np.array([[1,1,1],[0,0,0],[2,2,2], [0,0,0],[3,3,3], [0,0,0]])
arr = arr[~np.all(arr == 0, axis=1)]
print(arr)
'''
'''
test = np.empty((0,2, 3), dtype = np.float32)
test = np.insert(test, 0, arr[0], axis=0)
print(arr[1].shape)
#test = np.insert(test, 1, arr[1], axis=0)
test = np.vstack([arr[1]])
#print(test.shape)
print(test)
test = np.vstack([test,arr[0]])
print(test.shape)
'''

# for unit testing
def load_test(X_path, n_steps):
	
	file = open(X_path, 'r')
	X_ = np.array(
		[elem for elem in [
			row.split(',') for row in file
		]], 
		dtype=np.float32
	)
	file.close()
	blocks = int(len(X_) / n_steps)
	X_ = np.array(np.split(X_,blocks))

	return X_ 

def strip_five(arr):
    '''
    arg:
        np array of dimension (75, 98)
    return:
        sliced np array;
    function:
        remove the first 2 rows, and the last 3 rows;
    '''
    # hardcode the constant dimensions;
    assert arr.shape == (FRAMES, FEATURE), "np dimension should be (75, 98)"

    return(arr[2:72])

# remove all the zero frames;
def remove_zero_rows(arr):
    '''
    arg:
        np array of dimension (75, 98)
    return:
        np array with all the zero rows removed;
    '''
    return(arr[~np.all(arr == 0, axis=1)])

# this assumes the dimension = (70, 95);
def sub_sample(arr):
    '''
    arg:
        np array of dimension (70, 98)
    return:
        a tuple of two sub-sampled arrays;
    function:
        sub-sample the array at odd and even;
    '''
    return (arr[0::2], arr[1::2]) 

def make_up(arr, meet_size = 35):
    '''
    arg:
        np array;
        meet_size; the required size;
    return:
        a filled-up np array;    
    function:
        to fill up the array to meet a size requirement;
    '''

    # get the number of rows and columns;
    nrows = arr.shape[0]
    ncol = arr.shape[1]
    diff = meet_size - nrows
    
    # no point proceed;
    assert diff>0, "the number of rows is larger than the specified size"
    
    init = np.zeros((diff, ncol), dtype=np.float32)
    # append at the end;
    return(np.vstack([arr, init])) 

def random_sample(arr, size = 35):
    '''
    arg:
        np array;
        meet_size; the required size;
    return:
        a randomly sampled array  ;
    '''

    # get the number of rows;
    nrows = arr.shape[0]
    print(nrows)
    assert nrows > 0, "make sure the array is non-empty!!"
    # generate a list of random numbers within a range;
    # this function makes sure there's no replacement, which is important;
    # (IMPORTANT!!) sort it 
    numbers = sorted(random.sample(range(0, nrows), size))
    
    idx = np.array(numbers)
    # now, randomly sample the frames;
    return arr[idx,:]


# test driver;
if __name__ == '__main__':
    '''
    # txt file with alot of zero rows;
    TEST_PATH_02 = "C:\\Users\\yongw4\\Desktop\\X_train_02.txt"
    arrx = lstm.load_X(TEST_PATH_02)
    print(arrx.shape)
    test = arrx[0]
    print(test.shape)
    test = remove_zero_rows(strip_five(test))
    print(test.shape)
    '''
    
    '''
    naked = strip_five(test)
    print(naked.shape)

    (sub01, sub02) = sub_sample(naked)
    print(sub01.shape)
    print(sub02.shape)
    '''
    # txt file with 69 rows;
    TEST_PATH_03 = "C:\\Users\\yongw4\\Desktop\\X_train_03.txt"
    arrx = load_test(TEST_PATH_03, 69)
    test = arrx[0]
    print(test.shape)
    
    
    #test = make_up(test, meet_size = 75)
    test = random_sample(test)
    print(test.shape)
