import os
import io
import wave
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from groq import Groq
from gtts import gTTS

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def home():
    return {"status": "Robot Backend is Running Perfectly Without FFmpeg!"}


@app.post("/chat")
async def chat_with_robot(request: Request):
    # ESP32 থেকে আসা Raw PCM ডাটা রিসিভ করা

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
        # Groq Whisper দিয়ে ভয়েস থেকে টেক্সট করা

        wav_io.name = "input.wav"
        transcription = client.audio.transcriptions.create(
            file=wav_io,
            model="whisper-large-v3",
            language="bn"
        )
        user_text = transcription.text
        print(f"User: {user_text}")
       
        # Groq Llama-3 দিয়ে উত্তর জেনারেট করা

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "তুমি একটি বাংলাভাষী রোবট। খুব ছোট করে এক লাইনে বাংলায় উত্তর দাও।"},
                {"role": "user", "content": user_text}
            ]
        )
        robot_response = completion.choices.message.content
        print(f"Robot: {robot_response}")
       
        # gTTS দিয়ে সরাসরি WAV ডাটা জেনারেট করার টেকনিক
        # (এখানে আমরা gTTS দিয়ে টেক্সট জেনারেট করে সেটিকে সরাসরি byte stream এ নিয়ে রিড করব)

        tts = gTTS(text=robot_response, lang='bn')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
       
        # ট্রিক: যেহেতু আমরা ffmpeg বাদ দিয়েছি, gTTS এর MP3 ডাটাটিকে
        # সরাসরি রেসপন্স হিসেবে পাঠাবো।
        # দ্রষ্টব্য: এর জন্য আপনার ESP32 কোডে একটি ছোট পরিবর্তন লাগবে যা নিচে দেওয়া হলো।
        return StreamingResponse(mp3_fp, media_type="audio/mpeg")

       
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
