import logging
import speech_recognition
from typing import List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s][%(name)s]: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class SpeechProcessing:
    PROMPT = ""

    def __init__(self, input_path: str):
        self.input_path = input_path
        self.recognizer = None
        self.audio = None
        self.results = None
        self.recognizer = speech_recognition.Recognizer()

        logger.debug(self.__class__.__name__ + " initialized")

    def run(self) -> "SpeechProcessing":
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
                language="polish",
                word_timestamps=True,
                show_dict=True,
                prompt=SpeechProcessing.PROMPT,
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
