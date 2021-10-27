import os
import argparse




def main(args):
  for i in range(args.start_dir, args.num_dirs):
    log_dir = f"{args.root_dir}/{i}"
    print(f"Create dir in {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/test.txt", 'w') as f:
      f.write('test')
  pass



if __name__ == '__main__':
  """
  python3 template_lib/modelarts/scripts/make_log_dirs.py \
    --num_dirs 500 --start_dir 0 --root_dir /home/z50017127/user/logs
    
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_dirs', type=int, default=100)
  parser.add_argument('--start_dir', type=int, default=0)
  parser.add_argument('--root_dir', default="/tmp/logs")
  args = parser.parse_args()
  main(args)