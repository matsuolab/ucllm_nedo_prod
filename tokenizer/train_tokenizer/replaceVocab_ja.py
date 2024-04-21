import sentencepiece as spm

sp_en = spm.SentencePieceProcessor(model_file='JINIAC_V0_9_ja60000.model')
vocab_path = 'JINIAC_V0_9_ja60000.vocab'
with open(vocab_path, 'w', encoding='utf-8') as o:
  for id in range(sp_en.get_piece_size()):
    vocab = sp_en.id_to_piece(id)
    score = sp_en.GetScore(id)
    o.write(f'{vocab}\t{score}\n')
