from dataclasses import dataclass
from pathlib import Path

import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

NOUN_TAG = "NOUN"
ADJ_TAG = "ADJF"
ALLOWED_TAGS = {NOUN_TAG, ADJ_TAG}

PRONOUN_GRAMMEME = "Apro"

MIN_SCORE = 0.01


@dataclass
class WordInfo:
    word: str
    lemma: str
    part_of_speech: str | None
    gender: str | None
    number: str | None
    case: str | None
    score: float


def download_nltk_data() -> None:
    for package in ("punkt", "punkt_tab"):
        nltk.download(package, quiet=True)


def analyze_word(word: str, morph: pymorphy3.MorphAnalyzer) -> list[WordInfo]:
    variants = []

    for parse in morph.parse(word):
        if parse.tag.POS not in ALLOWED_TAGS:
            continue

        if parse.score < MIN_SCORE:
            continue

        if PRONOUN_GRAMMEME in parse.tag:
            continue

        variants.append(
            WordInfo(
                word=parse.word,
                lemma=parse.normal_form,
                part_of_speech=parse.tag.POS,
                gender=parse.tag.gender,
                number=parse.tag.number,
                case=parse.tag.case,
                score=parse.score,
            )
        )

    return variants


def agree(first: WordInfo, second: WordInfo) -> bool:
    if first.number is None or first.number != second.number:
        return False

    if first.case is None or first.case != second.case:
        return False

    if first.gender is None or second.gender is None:
        return True

    return first.gender == second.gender


def combination_bonus(first: WordInfo, second: WordInfo) -> float:
    if first.part_of_speech == ADJ_TAG and second.part_of_speech == NOUN_TAG:
        return 1.5

    if first.part_of_speech == ADJ_TAG or second.part_of_speech == ADJ_TAG:
        return 1.2

    return 1.0


def best_pair(
    first_variants: list[WordInfo],
    second_variants: list[WordInfo],
) -> tuple[WordInfo, WordInfo] | None:
    best = None
    best_score = 0.0

    for first in first_variants:
        for second in second_variants:
            if not agree(first, second):
                continue

            score = first.score * second.score * combination_bonus(first, second)

            if score > best_score:
                best = (first, second)
                best_score = score

    return best


def read_sentences(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")

    sentences = []

    for sentence in sent_tokenize(text, language="russian"):
        sentences.append(word_tokenize(sentence, language="russian"))

    return sentences


def split_by_punctuation(tokens: list[str]) -> list[list[str]]:
    groups: list[list[str]] = [[]]

    for token in tokens:
        if token.isalpha():
            groups[-1].append(token)
        else:
            groups.append([])

    return [group for group in groups if len(group) >= 2]


def main() -> None:
    download_nltk_data()

    morph = pymorphy3.MorphAnalyzer()
    path = Path(__file__).parent / "text.txt"

    for tokens in read_sentences(path):
        for group in split_by_punctuation(tokens):
            words = [analyze_word(token, morph) for token in group]

            for first_variants, second_variants in zip(words, words[1:]):
                pair = best_pair(first_variants, second_variants)

                if pair is None:
                    continue

                print(f"{pair[0].lemma} {pair[1].lemma}")


if __name__ == "__main__":
    main()