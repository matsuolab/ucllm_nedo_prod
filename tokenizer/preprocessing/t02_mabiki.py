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

    text_path = args.input
    mabiki_path = args.output

    input = 0
    flag =  0
    output = 0
    with open(mabiki_path, 'w', encoding='utf-8') as o:
      with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
          input += 1
          for word in tagger(line):
            if word.feature.lForm == None:
              flag += 1
          if flag == 0:
            o.write(line)
            output += 1
          flag = 0
          if input % 100000 == 0:
            print(f'input: {input}, output: {output}, rate: {output / input}')
    print(f'input: {input}, output: {output}, rate: {output / input}')


if __name__ == "__main__":
    main()
