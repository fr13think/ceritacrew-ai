# main.py
import os
import time
from fastapi import FastAPI, concurrency
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain_litellm import ChatLiteLLM
from dotenv import load_dotenv
from litellm.exceptions import RateLimitError

# Muat environment variables dari file .env
load_dotenv()

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="CrewAI Backend API",
    description="API untuk menjalankan agen AI CrewAI untuk pembuatan konten.",
    version="1.0.0"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Pydantic
class GenerationRequest(BaseModel):
    topic: str

# Inisialisasi LLM menggunakan ChatLiteLLM
llm = ChatLiteLLM(
    model="groq/llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# --- Definisikan Agen-Agen AI (Sudah diperbaiki menjadi lebih umum) ---
researcher = Agent(
    role='Peneliti Ahli',
    goal='Menemukan informasi mendalam, fakta unik, dan poin-poin kunci tentang topik "{topic}"',
    backstory=(
        "Anda adalah seorang peneliti digital yang bersemangat dalam menggali asal-usul, "
        "detail penting, dan cerita unik di balik topik apa pun. "
        "Anda sangat baik dalam menyajikan data dalam bentuk poin-poin yang jelas."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer = Agent(
    role='Penulis Konten Kreatif',
    goal='Menulis artikel blog yang menarik dan informatif tentang "{topic}" menggunakan informasi yang disediakan',
    backstory=(
        "Anda adalah seorang penulis konten yang mampu mengubah fakta-fakta menjadi cerita yang memikat. "
        "Gaya tulisan Anda santai, mudah dipahami, dan membuat pembaca tertarik."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- Endpoint API ---
@app.post("/generate")
async def generate_content(request: GenerationRequest):
    topic = request.topic
    
    research_task = Task(
        description=(
            "Lakukan riset tentang {topic}. Temukan asal-usulnya, "
            "karakteristik utamanya, "
            "serta satu fakta menarik yang jarang diketahui orang."
        ),
        expected_output=(
            "Sebuah laporan singkat dalam format poin-poin yang jelas. Contoh:\n"
            "- Asal/Sejarah: [Daerah asal atau sejarah singkat]\n"
            "- Karakteristik Utama: [Jelaskan poin-poin penting]\n"
            "- Fakta Unik: [Sebutkan fakta menarik]"
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Tulis sebuah draf artikel blog singkat (sekitar 300 kata) tentang {topic}. "
            "Gunakan HANYA informasi dari konteks yang diberikan oleh Peneliti Ahli. "
            "Buatlah judul yang menarik, pendahuluan yang singkat, isi yang menjelaskan poin-poin dari peneliti, "
            "dan kesimpulan yang kuat."
        ),
        expected_output=(
            "Sebuah draf artikel blog lengkap dalam format Markdown. Artikel harus dimulai dengan judul (contoh: '# Judul Artikel')."
        ),
        agent=writer,
        context=[research_task]
    )

    content_crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True 
    )

    try:
        # Menjalankan crew
        result = await concurrency.run_in_threadpool(
            content_crew.kickoff, inputs={'topic': topic}
        )
        return {"result": str(result)}
    except RateLimitError:
        # --- PENANGANAN RATE LIMIT ---
        # Jika kena rate limit, tunggu 7 detik dan coba lagi sekali.
        print("Rate limit tercapai. Menunggu 7 detik sebelum mencoba lagi...")
        time.sleep(7)
        result = await concurrency.run_in_threadpool(
            content_crew.kickoff, inputs={'topic': topic}
        )
        return {"result": str(result)}


@app.get("/")
def read_root():
    return {"message": "Selamat datang di CrewAI Content Generation API!"}
