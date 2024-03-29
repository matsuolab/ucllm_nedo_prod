import os
import re

in_path = '/persistentshare/storage/team_nakamura/member/horie/dataset/prefilter'
text_path = '/persistentshare/storage/team_nakamura/member/horie/dataset/text/ja_wiki.txt'

num = 0
with open(text_path, 'w', encoding='utf-8') as o:
#with open(textt_path, 'w', encoding='utf-8') as o:
  for root, _, files in os.walk(in_path):
  #for root, _, files in os.walk(test_path):
    for file in files:
      with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
        content = f.read()
        new_content = re.sub(r'^<[^>]*>$', '', content, flags=re.MULTILINE)
        e = re.sub(r'^\n', '', new_content, flags=re.MULTILINE)
        o.write(e)
        num += 1
        if num % 100 == 0:
          print(f'num: {num}')
# 最後の改行を削除する必要がある場合は、ファイルの末尾からそれを削除するロジックを追加することも可能ですが、
# 通常はそのままで問題ない場合が多いです。
print(f'num: {num}')