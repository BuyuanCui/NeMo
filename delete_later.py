from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.py import JapanesePhonemeTokenizer
from nemo.collections.tts.g2p.models.ja_jp_g2p import JapaneseG2p

JP_G2P  = JapaneseG2p(word_segmenter="janome",phoneme_dict="/home/alcui/JPG2P/NeMo_Jun18_JPG2P_ForkedRepo/scripts/tts_dataset_files/ja_JP/ja_JP_wordtoipa.txt")
