import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from groq import Groq
from gtts import gTTS

app = FastAPI()

# Groq API Key (আমরা এটি Render-এর ড্যাশবোর্ডে সেট করব)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def home():
    return {"status": "Robot Backend is Running!"}

@app.post("/chat")
async def chat_with_robot(file: UploadFile = File(...)):
    # ১. ESP32 থেকে আসা অডিও ফাইল সেভ করা
    input_audio_path = "input.wav"
    with open(input_audio_path, "wb") as f:
        f.write(await file.read())
   
    try:
        # ২. Groq Whisper দিয়ে অডিও থেকে বাংলা টেক্সট রূপান্তর
        with open(input_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="bn"
            )
        user_text = transcription.text
       
        # ৩. Groq Llama-3 দিয়ে বাংলায় ছোট উত্তর তৈরি
        system_prompt = "তুমি একটি বাংলাভাষী রোবট। ছোট এবং মিষ্টি করে এক লাইনে বাংলায় উত্তর দাও।"
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        robot_response = completion.choices.message.content
       
        # ৪. gTTS দিয়ে উত্তরটিকে বাংলা অডিওতে রূপান্তর
        output_audio_path = "output.mp3"
        tts = gTTS(text=robot_response, lang='bn')
        tts.save(output_audio_path)
       
        return FileResponse(output_audio_path, media_type="audio/mp3")
       
    except Exception as e:
        return {"error": str(e)}
