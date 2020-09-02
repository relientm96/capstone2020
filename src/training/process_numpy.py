import numpy as np
import LSTM_tools as lstm

# save and load the huge numpy array;
def npy_write(data, filename):
    np.save(filename, data)

def npy_read(filename):
    return np.load(filenam)

# test driver;
if __name__ == '__main__':
    txt_directory = "C:\\Users\\yongw4\\Desktop\\FATE\\txt-files\\speed-01"
    X_monstar, Y_monstar = lstm.patch_nparrays(txt_directory)
    npy_write(X_monstar, 'X_np')
    npy_write(Y_monstar, 'Y_np')