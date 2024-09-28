import logging
import ffmpeg
import speech_recognition


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
                language="polish",
                word_timestamps=True,
                show_dict=False,
                prompt=SpeechProcessingPipeline.PROMPT,
            )
        except speech_recognition.UnknownValueError:
            print("could not understand audio")
        except speech_recognition.RequestError as e:
            print("error; {0}".format(e))


if __name__ == "__main__":
    logger.info("Starting video to audio pipeline")

    input = "./data/HY_2024_film_02.mp4"
    temp = "data_processed/output.wav"

    VideoToAudioPipeline(input, temp).run()

    result = SpeechProcessingPipeline(temp).run().results
    print(result)
