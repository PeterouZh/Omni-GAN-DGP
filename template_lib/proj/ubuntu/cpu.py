import os

if __name__ == '__main__':
  """
  python -m template_lib.proj.ubuntu.cpu
  
  """
  run_str = """
  	cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c && echo cpus: &&  cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l && cat /proc/cpuinfo| grep "cpu cores"| uniq && echo threads: && cat /proc/cpuinfo| grep "processor"| wc -l
  """
  os.system(run_str)