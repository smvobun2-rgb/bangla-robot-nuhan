import os
import io
import wave
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from groq import Groq
from gtts import gTTS
from pydub import AudioSegment

app = FastAPI()


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def home():
    return {"status": "Robot Backend is Running Perfectly!"}

@app.post("/chat")
async def chat_with_robot(request: Request):
    # ESP32 থেকে সরাসরি Raw PCM ডাটা রিসিভ করা
    pcm_data = await request.body()
   
    # Raw PCM ডাটাকে WAV ফাইলে রূপান্তর (16kHz, 16-bit, Mono)
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm_data)
    wav_io.seek(0)
   
    try:
        # Groq Whisper দিয়ে ট্রান্সক্রিপশন
        # Groq এ ফাইল অবজেক্ট পাঠানোর জন্য নামসহ BytesIO পাস করতে হয়
        wav_io.name = "input.wav"
        transcription = client.audio.transcriptions.create(
            file=wav_io,

            model="whisper-large-v3",
            language="bn"
        )
        user_text = transcription.text
        print(f"User: {user_text}")
       
        # Groq Llama-3 দিয়ে উত্তর জেনারেট

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "তুমি একটি বাংলাভাষী রোবট। খুব ছোট করে এক লাইনে বাংলায় উত্তর দাও।"},

                {"role": "user", "content": user_text}
            ]
        )
        robot_response = completion.choices[0].message.content
        print(f"Robot: {robot_response}")
       
        # gTTS দিয়ে MP3 জেনারেট করা

        tts = gTTS(text=robot_response, lang='bn')
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        mp3_io.seek(0)
       
        # MP3 থেকে Raw PCM/WAV (16kHz, Mono) এ কনভার্ট করা (ESP32 এর জন্য)
        sound = AudioSegment.from_file(mp3_io, format="mp3")
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
       
        # শুধুমাত্র Raw PCM ডাটা স্ট্রিম আকারে পাঠানো (WAV Header ছাড়া, যাতে ESP32 সরাসরি বাজাতে পারে)
        pcm_output = sound.raw_data
       
        return StreamingResponse(io.BytesIO(pcm_output), media_type="audio/pcm")
       
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
