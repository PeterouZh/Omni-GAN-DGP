import sys
import unittest
import multiprocessing
import time


class TestingUnit(unittest.TestCase):

  def test_case1(self):
    def worker():
      """worker function"""
      print('Worker')
      return
    jobs = []
    for i in range(5):
      p = multiprocessing.Process(target=worker)
      jobs.append(p)
      p.start()

  def test_case2(self):
    def worker(num):
      """thread worker function"""
      print('Worker:', num)
      return
    jobs = []
    for i in range(5):
      p = multiprocessing.Process(target=worker, args=(i,))
      jobs.append(p)
      p.start()

  def test_case3(self):
    def worker():
      name = multiprocessing.current_process().name
      print(name, 'Starting')
      time.sleep(2)
      print(name, 'Exiting')

    def my_service():
      name = multiprocessing.current_process().name
      print(name, 'Starting')
      time.sleep(3)
      print(name, 'Exiting')

    service = multiprocessing.Process(name='my_service', target=my_service)
    worker_1 = multiprocessing.Process(name='worker 1', target=worker)
    worker_2 = multiprocessing.Process(target=worker)  # use default name

    worker_1.start()
    worker_2.start()
    service.start()

  def test_case4(self):
    def daemon():
      p = multiprocessing.current_process()
      print('Starting:', p.name, p.pid)
      sys.stdout.flush()
      time.sleep(2)
      print('Exiting :', p.name, p.pid)
      sys.stdout.flush()

    def non_daemon():
      p = multiprocessing.current_process()
      print('Starting:', p.name, p.pid)
      sys.stdout.flush()
      print('Exiting :', p.name, p.pid)
      sys.stdout.flush()

    d = multiprocessing.Process(name='daemon', target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon', target=non_daemon)
    n.daemon = False

    d.start()
    time.sleep(1)
    n.start()

  def test_case5(self):
    def daemon():
      print('Starting:', multiprocessing.current_process().name)
      time.sleep(2)
      print('Exiting :', multiprocessing.current_process().name)

    def non_daemon():
      print('Starting:', multiprocessing.current_process().name)
      print('Exiting :', multiprocessing.current_process().name)

    d = multiprocessing.Process(name='daemon', target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon', target=non_daemon)
    n.daemon = False

    d.start()
    time.sleep(1)
    n.start()

    d.join()
    n.join()

  def test_case6(self):
    def daemon():
      print('Starting:', multiprocessing.current_process().name)
      time.sleep(2)
      print('Exiting :', multiprocessing.current_process().name)

    def non_daemon():
      print('Starting:', multiprocessing.current_process().name)
      print('Exiting :', multiprocessing.current_process().name)

    d = multiprocessing.Process(name='daemon', target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon', target=non_daemon)
    n.daemon = False

    d.start()
    n.start()

    d.join(1)
    print('d.is_alive()', d.is_alive())
    n.join()

  def test_case7(self):

    class Worker(multiprocessing.Process):
      def run(self):
        args = self._args
        print('In %s' % self.name)
        return

    jobs = []
    for i in range(5):
      p = Worker(name='Worker-%d'%i, args=(i, i**2))
      jobs.append(p)
      p.start()
    for j in jobs:
      j.join()