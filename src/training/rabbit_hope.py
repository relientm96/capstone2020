# src - https://timber.io/blog/multiprocessing-vs-multithreading-in-python-what-you-need-to-know/
# src - https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes


import multiprocessing
import time
def shout(text):  
    return text.upper()  
  
def whisper(text):  
    return text.lower()  
  
def greet(func, dummy):  
    # storing the function in a variable  
    greeting = func("HELLO, yick")  
    print(greeting + dummy) 

if __name__ == '__main__':
    
    start_time = time.time()
    proc = []
    for i in [shout, whisper]:
        ## right here
        p = multiprocessing.Process(target=greet, args=(i, "dummy"))
        p.start()
        proc.append(p)
    for pp in proc:
        pp.join()
    print("--- %s seconds ---" % (time.time() - start_time))

    '''
    start_time = time.time()
    pool = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
    for i in range(0, 10):
        pool.apply_async(spawn2, args=(i,))
    pool.close()
    pool.join()
    print("--- %s seconds ---" % (time.time() - start_time))
    '''