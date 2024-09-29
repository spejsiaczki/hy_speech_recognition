from readability import Readability
import nltk


class GunningFog():
    def __init__(self, sample: str) -> None:
        nltk.download('punkt_tab')
        self._gf = None
        self.run(sample)

    def run(self, sample: str) -> None:
        t = self._multiply_sample(sample)
        r = Readability(t)
        self._gf = r.gunning_fog()

    @property
    def score(self) -> float:
        if not self._gf.score:
            self.run()
        return self._gf.score

    @property
    def grade_level(self) -> str:
        if not self._gf.grade_level:
            self.run()
        return self._gf.grade_level

    @property
    def grade_level_pl(self) -> str:
        score = self.score
        if 0 <= score < 7:
            return "język bardzo prosty," \
                "zrozumiały już dla uczniów szkoły podstawowej"
        elif 7 <= score < 10:
            return "język prosty," \
                " zrozumiały już dla uczniów gimnazjum"
        elif 10 <= score < 13:
            return "język dość prosty," \
                "zrozumiały już dla uczniów liceum"
        elif 13 <= score < 16:
            return "język dość trudny, " \
                "zrozumiały dla studentów studiów licencjackich"
        elif 16 <= score < 18:
            return "język trudny, " \
                "zrozumiały dla studentów studiów magisterskich"
        elif 18 <= score:
            return "język bardzo trudny, " \
                "zrozumiały dla magistrów i osób z wyższym wykształceniem"

    def _multiply_sample(self, sample: str):
        """
            Output sample must have more than 100 words.
            """
        no_words = len(sample.split())
        no_words_expected = 100
        multiply = (no_words_expected // no_words + 1)
        return sample * multiply
