import argparse
import datetime



def main():

  d1 = datetime.datetime.strptime(args.start, "%Y%m%d_%H%M%S")
  d2 = datetime.datetime.strptime(args.end, "%Y%m%d_%H%M%S")
  result = d2 - d1

  print(f'\nBetween {d1} and {d2}: \n {result}\n')
  print(f"total_seconds: {result.total_seconds()}")
  pass

if __name__ == '__main__':
  """
  python3 template_lib/proj/ubuntu/diff_dates.py \
    --start 20201220_173000 --end 20210114_101200
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--start', type=str, default="20201220_173000")
  parser.add_argument('--end', type=str, default="20210114_101200")

  args = parser.parse_args()
  main()

