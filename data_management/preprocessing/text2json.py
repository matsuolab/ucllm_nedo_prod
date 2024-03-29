import json

mabiki_path = '/persistentshare/storage/team_nakamura/member/horie/dataset/mabiki/ja_wiki_mabiki.txt'
json_path = '/persistentshare/storage/team_nakamura/member/horie/dataset/json/jawiki.jsonl'

input = 0
output = 0
length = 0
out = ""
with open(json_path, 'w', encoding='utf-8') as o:
#with open(jsont_path, 'w', encoding='utf-8') as o:
  with open(mabiki_path, 'r', encoding='utf-8') as f:
  #with open(textt_path, 'r', encoding='utf-8') as f:
    list = f.readlines()
    for line in list:
      input += 1
      length = length + len(line)
      if length >= 1000:
        output += 1
        sentence_dict = {'text': out}
        o.write(json.dumps(sentence_dict, ensure_ascii=False) + '\n')
        out = line
        length = len(line)
      else:        
        out = out + line
      if input % 10000 == 0:
        print(f'input: {input},output: {output}')
# 最後の改行を削除する必要がある場合は、ファイルの末尾からそれを削除するロジックを追加することも可能ですが、
# 通常はそのままで問題ない場合が多いです。
print(f'input: {input},output: {output}')