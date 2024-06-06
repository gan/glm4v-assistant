import base64
import os
from datetime import datetime
from threading import Lock, Thread
from time import sleep
from PIL import Image
import io
import cv2
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from speech_recognition import Microphone, Recognizer, UnknownValueError
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

import torch
import ChatTTS.ChatTTS as ChatTTS
from pydub import AudioSegment
from pydub.playback import play
import numpy as np

import logging

# 设置日志级别为WARNING，INFO级别的日志将不会显示
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ChatTTS").setLevel(logging.WARNING)
#logging.getLogger("qcloud_cos").setLevel(logging.WARNING)

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

# load ChatTTS model
chat = ChatTTS.Chat()
chat.load_models()

load_dotenv()

GLM_API_BASE = os.getenv("GLM_API_BASE")
GLM_API_KEY = os.getenv("GLM_API_KEY")

cos_region = os.getenv('COS_REGION')
cos_bucket_name = os.getenv('COS_BUCKET_NAME')
secret_id = os.getenv('COS_SECRET_ID')
secret_key = os.getenv('COS_SECRET_KEY')


config = CosConfig(Region=cos_region, SecretId=secret_id, SecretKey=secret_key, Token=None, Scheme='https')
client = CosS3Client(config)

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)
        image_url = self._upload_image(image)

        response = self.chain.invoke(
            {"prompt": prompt, "image_url": image_url},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    @staticmethod
    def _tts(response):
        wavs = chat.infer(response)
        # 假设生成的音频数据为单个音频
        audio_data = np.array(wavs[0] * 32767, dtype=np.int16)
        # 获取当前时间，并格式化为文件名
        file_name = f"./output/audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        # 创建一个AudioSegment实例
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=24000,
            sample_width=audio_data.dtype.itemsize,
            channels=1
        )
        # 保存音频数据到本地文件
        audio_segment.export(file_name, format="wav")
        # 播放音频
        play(audio_segment)

    # 上传图片图片
    def _upload_image(self, image):
        # 将图像转换为内存中的字节流
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        file_name = f"image-{os.urandom(4).hex()}.png"
        client.put_object(
            Bucket=cos_bucket_name,
            Body=byte_arr,
            Key=file_name,
            EnableMD5=False
        )
        return "https://"+cos_bucket_name+".cos."+cos_region+".myqcloud.com/" + file_name

    @staticmethod
    def _create_inference_chain(model):
        SYSTEM_PROMPT = """
        你是一个有眼睛的助手，我会发送图片给你，让你看到周围的景象，将使用用户提供的聊天历史和图片来回答其问题。
        不要提到“图片”这个单词，直接描述图片的内容，不要使用emojis，不要问用户问题。
        保持友好的态度。展示一些个性。不要太正式。
        用中文回复
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "{image_url}"
                            },
                        },
                    ],
                ),
            ]
        )
        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = WebcamStream().start()

model = ChatOpenAI(model="glm-4v", base_url=GLM_API_BASE, api_key=GLM_API_KEY)

assistant = Assistant(model)

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="small", language="chinese")
        assistant.answer(prompt, webcam_stream.read())

    except UnknownValueError:
        print("There was an error processing the audio.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)

#sleep(5)
#assistant.answer("你看到了什么", webcam_stream.read())
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.imshow("webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

webcam_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
