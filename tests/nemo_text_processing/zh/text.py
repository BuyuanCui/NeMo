import nemo_text_processing
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
inv=InverseNormalizer(lang='zh')
inv.inverse_normalize('一二三',verbose=True) #123

#string='一二三' # in tokenize_and_classify.py
#def tokenize(arg):
#    return [c for c in arg]
#tokenize mandarin, so instead of a one string, tokenize by character, one by one so it will work
#print(tokenize(string))

characters=''
for i in range ((0x4e00,0x9fff)):
    char=chr(i)
    if char not in mandarin_digits:
        characters=pynini.union(NEMO_SPACE + i)