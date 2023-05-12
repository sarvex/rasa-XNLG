import sys


def split_words(words):
  ret = []
  for word in words:
    ret.extend(iter(word))
  return ret


if __name__ == "__main__":
  # Usage: fn=test.q.zh.lc; cat data/xqg/$fn | python -u ./tools/zh_split_words.py > data/xqg-eval/$fn
  for line in sys.stdin:
    line = line.strip()
    words = line.split(" ")
    res = " ".join(split_words(words))
    print(f"{res}")  