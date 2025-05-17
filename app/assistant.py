import asyncio
import httpx
import json
import re
import string
import os
import typing
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from io import BytesIO
from starlette.websockets import WebSocketDisconnect, WebSocketState

from openai import AsyncOpenAI
from app.config import settings

DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-luna-en'
SYSTEM_PROMPT = """You are a helpful and enthusiastic assistant. Speak in a human, conversational tone.
Keep your answers as short and concise as possible, like in a conversation, ideally no more than 120 characters.
"""

client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class Assistant:
    def __init__(self, websocket, memory_size=10):
        self.websocket = websocket
        self.transcript_parts = []
        self.transcript_queue = asyncio.Queue()
        self.system_message = {'role': 'system', 'content': SYSTEM_PROMPT}
        self.chat_messages = []
        self.memory_size = memory_size
        self.httpx_client = httpx.AsyncClient()
        self.finish_event = asyncio.Event()
        self.file_id = None

    async def assistant_chat(self, messages: list[dict], model='gpt-4o'):
        """
        messages: list of {"role": "user"|"assistant", "content": str}
        """
        try:
            if not hasattr(self, 'assistant_id'):
                assistants = await openai_client.beta.assistants.list(limit=1)
                assistant = assistants.data[0]
                thread = await openai_client.beta.threads.create()
                self.assistant_id = assistant.id
                self.thread_id = thread.id

            msg = messages[-1]
            await openai_client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role=msg['role'],
                content=msg['content'],
                attachments=msg['attachments'] if 'attachments' in msg else None,
            )

            # stream_manager = openai_client.beta.threads.runs.stream(
            #     thread_id=self.thread_id,
            #     assistant_id=self.assistant_id,
            #     # event_handler=EventHandler(self.websocket),
            # )
            # async with stream_manager as stream:
            #     async for event in stream:
            #         print(event)
            #         if event.event == 'thread.message.delta':
            #             print(event.data)
            #             # await self.websocket.send_json(event.data)
            #         elif event.event == 'thread.message.completed':
            #             print(event.data.content)
            #             # await self.websocket.send_json(event.data)
            #     await stream.until_done()

            run = await openai_client.beta.threads.runs.create_and_poll(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id,
            )
            msgs = await openai_client.beta.threads.messages.list(
                thread_id=self.thread_id
            )
            assistant_replies = [
                m for m in msgs.data
                if m.role == 'assistant'
            ]
            full_response = ''
            for reply in assistant_replies[0:1]:
                text = reply.content[0].text.value
                full_response += text

            return full_response
        
        except Exception as error:
            print(error)
            return str(error)

    async def upload_pdf(self, file_path, file_name):
        """Upload a PDF file to OpenAI and return the file ID"""
        with open(file_path, 'rb') as f:
            file = await openai_client.files.create(
                file=f,
                purpose='user_data'
            )
            self.file_id = file.id
            return file.id
    
    def should_end_conversation(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip().lower()
        return re.search(r'\b(goodbye|bye)\b$', text) is not None
    
    async def text_to_speech(self, text):
        audio_stream = client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=settings.ELEVENLABS_VOICE_ID,
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
            language_code="fr",
        )
        for chunk in audio_stream:
            await self.websocket.send_bytes(chunk)
    
    async def transcribe_audio(self):
        buffer = bytearray()
        while not self.finish_event.is_set():
            message = await self.websocket.receive()
            try:
                if "bytes" in message:
                    buffer.extend(message["bytes"])

                    if len(buffer) > 3:
                        bio = BytesIO(buffer)
                        transcription = client.speech_to_text.convert(
                            file=bio,
                            model_id="scribe_v1",
                            tag_audio_events=True,
                            language_code="fr",
                            diarize=False,
                        )
                        text = transcription["text"]
                        await self.transcript_queue.put({
                            "type": "speech_final", "content": text
                        })
                        buffer.clear()

                elif 'text' in message:
                    try:
                        json_data = message["text"]
                        data = json.loads(json_data)
                        if data.get("type") == "attach":
                            file_id = await self.add_pdf_context(data.get("content"))
                            self.file_id = file_id
                        else:
                            await self.transcript_queue.put({'type': 'speech_final', 'content': data.get("content")})
                    except Exception as error:
                        print(f"Error processing JSON message: {str(error)}")
                else:
                    break

            except Exception as error:
                print(f"Error in transcribe_audio: {str(error)}")
    
    async def manage_conversation(self):
        try:
            while not self.finish_event.is_set():
                transcript = await self.transcript_queue.get()
                if transcript['type'] == 'speech_final':
                    if self.should_end_conversation(transcript['content']):
                        self.finish_event.set()
                        await self.websocket.send_json({'type': 'finish'})
                        break
                    if self.file_id:
                        self.chat_messages.append({
                            'role': 'user',
                            'content': transcript['content'],
                            'attachments': [
                                {
                                    'tools': [{"type": "code_interpreter"}],
                                    'file_id': self.file_id
                                }
                            ]
                        })
                        self.file_id = None
                    else:
                        self.chat_messages.append({'role': 'user', 'content': transcript['content']})

                    response = await self.assistant_chat(
                        self.chat_messages[-self.memory_size:]
                        # [self.system_message] + self.chat_messages[-self.memory_size:]
                    )
                    self.chat_messages.append({'role': 'assistant', 'content': response})
                    await self.websocket.send_json({'type': 'assistant', 'content': response})
                    await self.text_to_speech(response)
                else:
                    await self.websocket.send_json(transcript)
        except Exception as error:
            print(str(error))
    
    async def add_pdf_context(self, file_path):
        """Add a PDF file as context for the conversation"""
        file_id = await self.upload_pdf(file_path, os.path.basename(file_path))
        return file_id

    async def run(self):
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.transcribe_audio())
                tg.create_task(self.manage_conversation())
        except* WebSocketDisconnect:
            print('Client disconnected')
        finally:
            await self.httpx_client.aclose()
            if self.websocket.client_state != WebSocketState.DISCONNECTED:
                await self.websocket.close()
