import argparse
from ast import parse
from fugashi import Tagger

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True) 
    parser.add_argument("--output", type=str, required=True)     
    args = parser.parse_args()
    print(f"{args = }")
    return args


def main():
    args = parse_arguments()  
    tagger = Tagger('-Owakati')

    mabiki_path = args.input
    wakachi_path = args.output

    num = 0
    wakachi = ""
    with open(wakachi_path, 'w', encoding='utf-8') as o:
      with open(mabiki_path, 'r', encoding='utf-8') as f:
        for line in f:
          num += 1
          for word in tagger(line):
            wakachi = wakachi + word.surface + '||||'
          o.write(wakachi + '\n')
          wakachi = ""
          if num % 100000 == 0:
            print(f'num: {num}')
    print(f'num: {num}')


if __name__ == "__main__":
    main()
