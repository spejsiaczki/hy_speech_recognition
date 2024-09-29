from langdetect import detect_langs, LangDetectException


class LanguageDetection:
    FILANAME_LANG_SET = "./lang_set.txt"

    def __init__(self):
        self.lang_set = self._load_lang_set()

    def _norm_word(self, word: str) -> str:
        return word.replace(",", "").replace(".", "").lower().strip()

    def _load_lang_set(self):
        with open(self.FILANAME_LANG_SET, "r") as f:
            return set(self._norm_word(w) for w in f.read().splitlines())

    def detect_polish(self, text: str) -> bool:
        """
        Returns a list of detected languages with confidence scores
        """

        word = self._norm_word(text)
        if word in self.lang_set:
            return True

        if any(ch in text for ch in "ęóąśłżźćń"):
            return True

        try:
            probs = [(lang.lang, lang.prob) for lang in detect_langs(text)]
            if probs[0] == "pl":
                return True
            else:
                return False
        except LangDetectException:
            return False

    @property
    def languages(self):
        return self.possible_languages
