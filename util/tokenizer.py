import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import re
from typing import List
import jieba.posseg as pseg

### STOP WORDS
stopwords_zh_path = os.path.abspath(os.path.dirname(__file__) + '/../data/stopwords_zh.txt')
STOPWORDS_ZH = set()
with open(stopwords_zh_path, "r") as f:
    for line in f:
        STOPWORDS_ZH.add(line.strip())
        
stopwords_en_path = os.path.abspath(os.path.dirname(__file__) + '/../data/stopwords_en.txt')
STOPWORDS_EN = set()
with open(stopwords_en_path, "r") as f:
    for line in f:
        STOPWORDS_EN.add(line.strip())


### TOKENIZATION ###
english_tokenizer = re.compile(r'\b\w+\b')
FILTERED_TAGS = {'r', 'u', 'w', 'x', 'y'}  # 代词、助词、标点、非语素字、语气词

def tokenize_text(text: str, lang='zh') -> List[str]:
    if lang == 'zh':
        # 使用 jieba.posseg 对文本进行分词和词性标注
        tokens = pseg.cut(text)
        
        # 过滤条件：不在停用词中、长度大于1且不属于过滤的词性标签
        result = [
            token.word for token in tokens 
            if token.word not in STOPWORDS_ZH and len(token.word) > 1 and token.flag not in FILTERED_TAGS
        ]
        return result
        # return [token for token in jieba.cut(text, cut_all=False) if token not in STOPWORDS_ZH and len(token) > 1]
    else:
        return [token for token in english_tokenizer.findall(text) if token not in STOPWORDS_EN and len(token) > 2]
