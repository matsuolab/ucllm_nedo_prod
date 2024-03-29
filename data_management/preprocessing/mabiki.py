from fugashi import Tagger

tagger = Tagger('-Owakati')

text_path = '/persistentshare/storage/team_nakamura/member/horie/dataset/text/ja_wiki.txt'
mabiki_path = '/persistentshare/storage/team_nakamura/member/horie/dataset/mabiki/ja_wiki_mabiki.txt'

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