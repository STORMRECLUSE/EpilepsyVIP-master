__author__ = 'Chris'

import multiprocessing
import time

def worker():
    name = multiprocessing.current_process().name
    print name + " is going now!"
    time.sleep(5)
    return

if __name__ == '__main__':
    job1 = multiprocessing.Process(target=worker)
    job2 = multiprocessing.Process(target=worker)
    job3 = multiprocessing.Process(target=worker)

    job2.start()
    job1.start()
    job3.start()