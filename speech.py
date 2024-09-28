import logging
import ffmpeg
import speech_recognition
from readability import Readability
from typing import List, Tuple
import nltk
from langdetect import detect_langs, LangDetectException


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s][%(name)s]: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class VideoToAudioPipeline:
    def __init__(self,
                 input_path: str,
                 output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        logger.debug(self.__class__.__name__ + " initialized")

    def run(self) -> "SpeechProcessingPipeline":
        logger.debug(
            f"Running video to audio conversion from {self.input_path} to {self.output_path}")
        self._preprocess_audio(self.input_path, self.output_path)
        return self

    def _preprocess_audio(self,
                          input_file: str,
                          output_file: str) -> None:
        try:
            ffmpeg \
                .input(input_file) \
                .output(output_file,
                        acodec='pcm_s16le',
                        ar=44100,
                        ) \
                .overwrite_output() \
                .run(quiet=True)

            logger.debug(
                f"Audio extracted from {input_file}"
                f"and saved to {output_file}")
        except ffmpeg.Error:
            logger.exception("Error occurred while extracting audio")


class SpeechProcessingPipeline:
    PROMPT = ""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.recognizer = None
        self.audio = None
        self.results = None
        self.recognizer = speech_recognition.Recognizer()

        logger.debug(self.__class__.__name__ + " initialized")

    def run(self) -> "SpeechProcessingPipeline":
        self._read_audio()
        self._transcribe()
        return self

    def _read_audio(self) -> str:
        try:
            with speech_recognition.AudioFile(self.input_path) as src:
                self.audio = self.recognizer.record(src)
        except FileNotFoundError:
            logger.exception(f"File not found: {self.input_path}")

    def _transcribe(self) -> str:
        try:
            self.results = self.recognizer.recognize_whisper(
                model="small",
                audio_data=self.audio,
                # language="polish",
                word_timestamps=True,
                show_dict=True,
                prompt=SpeechProcessingPipeline.PROMPT,
            )
        except speech_recognition.UnknownValueError:
            print("could not understand audio")
        except speech_recognition.RequestError as e:
            print("error; {0}".format(e))

    def get_word_timestamps(self) -> List[Tuple[str, float, float]]:
        timestamps = []
        for segment in self.results["segments"]:
            for word in segment["words"]:
                text = word["word"]
                timestamps.append((text, word["start"], word["end"]))
        return timestamps

    def get_text(self) -> str:
        return "".join(self.get_words())

    def get_words(self) -> List[str]:
        words = []
        for segment in self.results["segments"]:
            for word in segment["words"]:
                words.append(word["word"])
        return words

    def get_pause_timestamps(self) -> List[Tuple[float, float]]:
        pauses = []
        word_timestamps = self.get_word_timestamps()
        for first, second in zip(word_timestamps, word_timestamps[1:]):
            t_start = first[2]
            t_end = second[1]
            if t_end - t_start > 0.01:
                pauses.append((t_start, t_end))
        return pauses


class GunningFog():
    def __init__(self, sample: str) -> None:
        nltk.download('punkt_tab')
        self._gf = None
        self.run(sample)

    def run(self, sample: str) -> None:
        t = self._multiply_sample(sample)
        r = Readability(t)
        self._gf = r.gunning_fog()

    @ property
    def score(self) -> float:
        if not self._gf.score:
            self.run()
        return self._gf.score

    @ property
    def grade_level(self) -> str:
        if not self._gf.grade_level:
            self.run()
        return self._gf.grade_level

    @ property
    def grade_level_pl(self) -> str:
        score = self.score
        if 0 <= score < 7:
            return "język bardzo prosty,"
            "zrozumiały już dla uczniów szkoły podstawowej"
        elif 7 <= score < 10:
            return "język prosty,"
            " zrozumiały już dla uczniów gimnazjum"
        elif 10 <= score < 13:
            return "język dość prosty,"
            "zrozumiały już dla uczniów liceum"
        elif 13 <= score < 16:
            return "język dość trudny, "
            "zrozumiały dla studentów studiów licencjackich"
        elif 16 <= score < 18:
            return "język trudny, "
            "zrozumiały dla studentów studiów magisterskich"
        elif 18 <= score:
            return "język bardzo trudny, "
            "zrozumiały dla magistrów i osób z wyższym wykształceniem"

    def _multiply_sample(self, sample: str):
        """
            Output sample must have more than 100 words.
            """
        no_words = len(sample.split())
        no_words_expected = 100
        multiply = (no_words_expected // no_words + 1)
        return sample * multiply


class LanguageDetection:
    def __init__(self, text: str):
        self.text = text
        self.possible_languages = self._detect_language()

    def _detect_language(self):
        """
        Returns a list of detected languages with confidence scores
        """
        if any(ch in self.text for ch in "ęóąśłżźćń"):
            return [("pl", 1.0)]

        if len(self.text) < 5:
            return []

        try:
            return [(lang.lang, lang.prob) for lang in detect_langs(self.text)]
        except LangDetectException:
            return []

    @property
    def languages(self):
        return self.possible_languages


if __name__ == "__main__":
    logger.info("Starting video to audio pipeline")

    input = "./data/HY_2024_film_17.mp4"
    temp = "data_processed/output.wav"

    VideoToAudioPipeline(input, temp).run()

    speach_processing = SpeechProcessingPipeline(temp).run()
    print(speach_processing.results)
    print(speach_processing.get_word_timestamps())
    print(speach_processing.get_text())
    print(speach_processing.get_pause_timestamps())

    for word in speach_processing.get_words():
        lang_detect = LanguageDetection(word)
        print(word, lang_detect.languages)

    TEXT = """Mam ciągle w uszach głos, twój ciepły głos i oczy ciągle ciebie pełne mam, a już pod stopą moją dudni, dudni most; przez wiatr, przez mróz, przez słońce i przez zieleń maszeruję."""

    gf = GunningFog(TEXT)
    print(gf.score)
    print(gf.grade_level)
    print(gf.grade_level_pl)
