import sys

SYMBOL = '▁'

newVocab = set()
newVocabList = []

for line in open(sys.argv[1]):
    line = line.rstrip()
    if line == '':
      continue
    else:
      token, score = line.split('\t')
      if token == SYMBOL:
          pass
      elif token.startswith(SYMBOL):
          token = token.replace(SYMBOL, '')
      if token not in newVocab:
          newVocab.add(token)
          newVocabList.append(token + '\t' + score)

for line in newVocabList:
    print(line)

#print('size of new vocab:', len(newVocab))