import asyncio
import httpx
import json
import re
import string
import os
import typing
from starlette.websockets import WebSocketDisconnect, WebSocketState
from deepgram import (
    DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
)
from openai import AsyncOpenAI
from app.config import settings

DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-luna-en'
SYSTEM_PROMPT = """You are a helpful and enthusiastic assistant. Speak in a human, conversational tone.
Keep your answers as short and concise as possible, like in a conversation, ideally no more than 120 characters.
"""

deepgram_config = DeepgramClientOptions(options={'keepalive': 'true'})
deepgram = DeepgramClient(settings.DEEPGRAM_API_KEY, config=deepgram_config)
dg_connection_options = LiveOptions(
    model='nova-2',
    language='en',
    # Apply smart formatting to the output
    smart_format=True,
    # To get UtteranceEnd, the following must be set:
    interim_results=True,
    utterance_end_ms='1000',
    vad_events=True,
    # Time in milliseconds of silence to wait for before finalizing speech
    endpointing=500,
)
openai_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    # http_client=httpx.AsyncClient(
    #     proxy=settings.OPENAI_PROXY
    # )
)
# groq = AsyncGroq(api_key=settings.GROQ_API_KEY)

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
        self.is_first_message = True
    
    # async def assistant_chat(self, messages, model='llama-4-scout-17b-16e-instruct'):
    #     res = await groq.chat.completions.create(messages=messages, model=model)
    #     return res.choices[0].message.content
    
    async def assistant_chat(self, messages, model='gpt-4o'):
        try:
            res = await openai_client.responses.create(input=messages, model=model)
            return res.output_text
        except Exception as error:
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
        headers = {
            'Authorization': f'Token {settings.DEEPGRAM_API_KEY}',
            'Content-Type': 'application/json'
        }
        async with self.httpx_client.stream(
            'POST', DEEPGRAM_TTS_URL, headers=headers, json={'text': text}
        ) as res:
            async for chunk in res.aiter_bytes(1024):
                await self.websocket.send_bytes(chunk)
    
    async def transcribe_audio(self):
        async def on_message(self_handler, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) == 0:
                    return
                if result.is_final:
                    self.transcript_parts.append(sentence)
                    await self.transcript_queue.put({'type': 'transcript_final', 'content': sentence})
                    if result.speech_final:
                        full_transcript = ' '.join(self.transcript_parts)
                        self.transcript_parts = []
                        await self.transcript_queue.put({'type': 'speech_final', 'content': full_transcript})
                else:
                    await self.transcript_queue.put({'type': 'transcript_interim', 'content': sentence})
            except Exception as error:
                raise Exception(str(error))
        
        async def on_utterance_end(self_handler, utterance_end, **kwargs):
            if len(self.transcript_parts) > 0:
                full_transcript = ' '.join(self.transcript_parts)
                self.transcript_parts = []
                await self.transcript_queue.put({'type': 'speech_final', 'content': full_transcript})

        dg_connection = deepgram.listen.asynclive.v('1')
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        if await dg_connection.start(dg_connection_options) is False:
            raise Exception('Failed to connect to Deepgram')
        
        try:
            while not self.finish_event.is_set():
                message = await self.websocket.receive()

                if 'bytes' in message:
                    audio_data = typing.cast(bytes, message["bytes"])
                    await dg_connection.send(audio_data)
                elif 'text' in message:
                    try:
                        json_data = message["text"]
                        data = json.loads(json_data)
                        await self.transcript_queue.put({'type': 'speech_final', 'content': data.get("content")})
                    except Exception as error:
                        print(f"Error processing JSON message: {str(error)}")
                else:
                    break
        except Exception as error:
            print(f"Error in transcribe_audio: {str(error)}")

        finally:
            await dg_connection.finish()
    
    async def manage_conversation(self):
        try:
            while not self.finish_event.is_set():
                transcript = await self.transcript_queue.get()
                if transcript['type'] == 'speech_final':
                    if self.should_end_conversation(transcript['content']):
                        self.finish_event.set()
                        await self.websocket.send_json({'type': 'finish'})
                        break
                    if self.is_first_message and self.file_id:
                        self.chat_messages.append({
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'input_file',
                                    'file_id': self.file_id
                                }, {
                                    'type': 'input_text',
                                    'text': transcript['content']
                                }
                            ]
                        })
                        self.is_first_message = False
                    else:
                        self.chat_messages.append({'role': 'user', 'content': transcript['content']})

                    response = await self.assistant_chat(
                        [self.system_message] + self.chat_messages[-self.memory_size:]
                    )
                    self.chat_messages.append({'role': 'assistant', 'content': response})
                    await self.websocket.send_json({'type': 'assistant', 'content': response})
                    await self.text_to_speech(response)
                else:
                    await self.websocket.send_json(transcript)
        except Exception as error:
            raise Exception(str(error))
    
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
