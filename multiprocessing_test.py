import multiprocessing
import time
import math

## for phase screen multiprocessing use POOLs and a multiprocessing queue for the results
  
def count(n):
    return math.sqrt(n)
  
if __name__ == "__main__":
    n = 1e9
    N = 10000

    # simple method
    start = time.time()
    for _ in range(N):
        count(n)
    end = time.time()
    print(end-start)

    # processes
    # start = time.time()
    # proc = []
    # for _ in range(N):
    #     proc.append(multiprocessing.Process(target=count, args=(n, )))
    #     proc[-1].start()
    # for i in range(N):
    #     proc[i].join()
    # end = time.time()
    # print(end-start)

    with multiprocessing.Pool() as pool:
        # pools
        start = time.time()
        res = pool.map(func=count, iterable=[n]*N)
        end = time.time()
        print(end-start)

        # pool unordered
        start = time.time()
        res = pool.imap_unordered(func=count, iterable=[n]*N)
        end = time.time()
        print(end-start)

        # pool async
        start = time.time()
        res = [pool.apply_async(count, (n,)) for i in range(N)]
        [r.get(timeout=1) for r in res]
        end = time.time()
        print(end-start)


