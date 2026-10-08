import re
from pathlib import Path

import gensim

MODEL_PATH = Path(__file__).parent / "cbow.txt"

POSITIVE = ["газик_NOUN", "таксист_NOUN"]
NEGATIVE = ["шофер_NOUN"]

TOP_NOUNS = 10
RAW_TOPN = 200

NOUN_PATTERN = re.compile("(.*)_NOUN")


def main() -> None:
    model = gensim.models.KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=False)

    dist = model.most_similar(positive=POSITIVE, negative=NEGATIVE, topn=RAW_TOPN)

    number = 0

    for token, score in dist:
        match = NOUN_PATTERN.match(token)

        if match is None:
            continue

        number += 1
        print(f"{number:2}. {match.group(1)}")

        if number == TOP_NOUNS:
            break


if __name__ == "__main__":
    main()