import argparse
from ast import parse
import os
import re
import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True) 
    parser.add_argument("--input_base", type=str, required=True) 
    parser.add_argument("--output_base", type=str, required=True)     
    args = parser.parse_args()
    print(f"{args = }")
    return args


def main():
    args = parse_arguments()

    lang = args.language
    in_path = args.input_base + lang
    json_path = args.output_base + lang + '_wiki.jsonl'

    os.makedirs(args.output_base, exist_ok=True)
    
    num = 0
    with open(json_path, 'w', encoding='utf-8', newline='\n') as o:
      for root, _, files in os.walk(in_path):
        for file in files:
          if lang == "ja" or (lang == "en" and file.endswith('9')):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
              content = f.read()
              new_content = re.sub(r'^<[^>]*>$', '', content, flags=re.MULTILINE)
              e = re.sub(r'^\n', '', new_content, flags=re.MULTILINE)
              for line in e.split('\n'):
                if len(line.strip()) > 0:
                  sentence_dict = {'text': line}
                  o.write(json.dumps(sentence_dict, ensure_ascii=False) + '\n')
                  num += 1
                  if num % 100000 == 0:
                    print(f'num: {num}')
    # 最後の改行を削除する必要がある場合は、ファイルの末尾からそれを削除するロジックを追加することも可能ですが、
    # 通常はそのままで問題ない場合が多いです。
    print(f'num: {num}')


if __name__ == "__main__":
    main()
