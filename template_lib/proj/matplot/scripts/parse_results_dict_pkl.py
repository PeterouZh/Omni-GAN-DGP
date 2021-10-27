import argparse
import pickle


def main():
  with open(args.pkl, 'rb') as f:
    loaded_dict = pickle.load(f)
  for dict_index, data_dict in loaded_dict.items():
    print(f"dict index: {dict_index}")
    for data_index in data_dict.keys():
      print(f"\t data index: {data_index}")

  pass


if __name__ == '__main__':
  """
  python template_lib/proj/matplot/scripts/parse_results_dict_pkl.py --pkl 
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('--pkl', type=str)

  args = parser.parse_args()
  main()