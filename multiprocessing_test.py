import multiprocessing
import time
import math

## for phase screen multiprocessing use POOLs and a multiprocessing queue for the results
  
def count(n):
    for i in range(1,int(abs(n))+1):
        math.sqrt(i)
  
if __name__ == "__main__":
    target = 2.5e7 # simple method wins if target < 1e6

    # simple method
    start = time.time()
    count(target)
    count(target)
    count(target)
    end = time.time()
    print(end-start)

    # processes
    start = time.time()
    p1 = multiprocessing.Process(target=count, args=(target, ))
    p2 = multiprocessing.Process(target=count, args=(target, ))  
    p3 = multiprocessing.Process(target=count, args=(target, ))  
    p1.start()
    p2.start()  
    p3.start()  
    p1.join()
    p2.join()
    p3.join()
    end = time.time()
    print(end-start)

    # pools
    start = time.time()
    pool = multiprocessing.Pool()
    pool.map(func=count, iterable=[target, target, target])
    pool.close()
    end = time.time()
    print(end-start)

