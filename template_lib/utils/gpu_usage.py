import subprocess
import unittest
import pprint


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    mem_used = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    mem_total = subprocess.check_output(
      [
        'nvidia-smi', '--query-gpu=memory.total',
        '--format=csv,nounits,noheader'
      ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = ['%s/%s MB'%(used, total) for used, total
                  in zip(mem_used.strip().split('\n'),
                         mem_total.strip().split('\n'))]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    gpu_memory_map = pprint.pformat(gpu_memory_map)
    return gpu_memory_map


class TestingUnit(unittest.TestCase):

  def test_get_gpu_memory_map(self):
    gpu_str = get_gpu_memory_map()
    print(gpu_str)