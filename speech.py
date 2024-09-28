import logging
import ffmpeg
import speech_recognition
from readability import Readability
from typing import List, Tuple
import nltk


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
        return "".join(
            segment["text"] for segment in self.results["segments"])


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
            return "Wczesna szkoła podstawowa"
        elif 7 <= score < 10:
            return "Późna szkoła podstawowa"
        elif 10 <= score < 13:
            return "Szkoła średnia"
        elif 13 <= score < 16:
            return "Studia licencjackie"
        elif 16 <= score < 18:
            return "Studia magisterskie"
        elif 18 <= score:
            return "Doktorat"

    def _multiply_sample(self, sample: str):
        """
        Output sample must have more than 100 words.
        """
        no_words = len(sample.split())
        no_words_expected = 100
        multiply = (no_words_expected // no_words + 1)
        return sample * multiply


if __name__ == "__main__":
    logger.info("Starting video to audio pipeline")

    input = "./data/HY_2024_film_02.mp4"
    temp = "data_processed/output.wav"

    VideoToAudioPipeline(input, temp).run()

    speach_processing = SpeechProcessingPipeline(temp).run()
    print(speach_processing.results)
    print(speach_processing.get_word_timestamps())
    print(speach_processing.get_text())

    gf = GunningFog(speach_processing.get_text())
    print(gf.score)
    print(gf.grade_level)
    print(gf.grade_level_pl)
