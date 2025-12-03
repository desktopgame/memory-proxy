import MeCab
import ipadic
import re

# MeCab tagger for Japanese morphological analysis
_mecab = MeCab.Tagger(ipadic.MECAB_ARGS)
_mecab.parse("")  # GCクラッシュ回避の定型

# Part-of-speech patterns to extract as keywords (IPAdic format)
# Format: "品詞,品詞細分類1,品詞細分類2,品詞細分類3"
EXTRACT_POS_PATTERNS = [
    "名詞,一般",
    "名詞,固有名詞",
    "名詞,サ変接続",
    "名詞,形容動詞語幹",
    "名詞,代名詞",  # 私, 僕, あなた, etc.
]
# Minimum keyword length
MIN_KEYWORD_LENGTH = 1


def extract_keywords_mecab(text: str) -> list[str]:
    """
    Extract keywords from text using MeCab morphological analysis.
    Also handles English words.
    """
    keywords = set()
    
    # Parse with MeCab
    node = _mecab.parseToNode(text)
    while node:
        # Skip BOS/EOS nodes
        if node.stat not in (MeCab.MECAB_BOS_NODE, MeCab.MECAB_EOS_NODE):
            surface = node.surface
            feature = node.feature  # "品詞,品詞細分類1,品詞細分類2,..."
            
            # Check if this part of speech should be extracted
            for pattern in EXTRACT_POS_PATTERNS:
                if feature.startswith(pattern):
                    # Get base form (7th element in IPAdic) if available
                    parts = feature.split(",")
                    base_form = parts[6] if len(parts) > 6 and parts[6] != "*" else surface
                    
                    if len(base_form) >= MIN_KEYWORD_LENGTH:
                        keywords.add(base_form.lower())
                    break
        
        node = node.next
    
    # Also extract English words (alphanumeric sequences)
    english_words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", text)
    for word in english_words:
        if len(word) >= MIN_KEYWORD_LENGTH:
            keywords.add(word.lower())
    
    return list(keywords)
