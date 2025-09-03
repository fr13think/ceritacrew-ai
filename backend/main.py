# main.py
# Dependencies:
#   pip install -U fastapi uvicorn python-dotenv crewai litellm

import os
import time
from fastapi import FastAPI, concurrency, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from litellm.exceptions import RateLimitError

# Muat .env
load_dotenv()

# Ambil API key NVIDIA NIM (OpenAI-compatible)
NIM_API_KEY = (
    os.getenv("NIM_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
).strip().strip('"').strip("'")
if not NIM_API_KEY:
    raise RuntimeError("Set NIM_API_KEY / NVIDIA_API_KEY / OPENAI_API_KEY di .env")

# Model dan base URL (prefix 'openai/' PENTING untuk LiteLLM)
# Contoh: openai/meta/llama-3.1-8b-instruct (cepat), openai/meta/llama-3.3-70b-instruct (lebih berkualitas)
MODEL_NAME = os.getenv("NIM_MODEL", "openai/meta/llama-3.1-8b-instruct")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

app = FastAPI(
    title="CrewAI Backend API",
    description="API untuk menjalankan agen AI CrewAI untuk pembuatan konten.",
    version="1.0.0"
)

# CORS: izinkan file:// (Origin: null) dan origin lain saat dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging sederhana
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f">>> {request.method} {request.url} | Origin={request.headers.get('origin')}")
    resp = await call_next(request)
    print(f"<<< {request.method} {request.url.path} -> {resp.status_code}")
    return resp

class GenerationRequest(BaseModel):
    topic: str

# LLM per agent:
# - Researcher: akurat/tenang (rendah temperature)
llm_researcher = LLM(
    model=MODEL_NAME,
    api_key=NIM_API_KEY,
    base_url=NIM_BASE_URL,
    temperature=0.2,
    top_p=0.7,
    max_tokens=512,
)

# - Writer: ekspresif/kreatif (lebih tinggi temperature + presence_penalty)
llm_writer = LLM(
    model=MODEL_NAME,
    api_key=NIM_API_KEY,
    base_url=NIM_BASE_URL,
    temperature=0.9,
    top_p=0.95,
    presence_penalty=0.6,  # dorong variasi ide
    frequency_penalty=0.1, # kurangi pengulangan
    max_tokens=768,
)

print(f"[Startup] Researcher LLM={MODEL_NAME}, Writer LLM={MODEL_NAME} | base_url={NIM_BASE_URL}")

# --- Agen (dipertahankan seperti versi awal, dengan LLM masing-masing) ---
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
    llm=llm_researcher
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
    llm=llm_writer
)

@app.post("/generate")
async def generate_content(request: GenerationRequest):
    topic = request.topic
    
    research_task = Task(
        description=(
            "Lakukan riset tentang {topic}. Temukan asal-usulnya, karakteristik utamanya, "
            "serta satu fakta menarik yang jarang diketahui orang. "
            "Jangan berikan daftar sumber atau bibliografi di akhir jawaban."
        ),
        expected_output=(
            "Sajikan laporan ringkas berbentuk bullet yang padat dan faktual:\n"
            "- Asal/Sejarah: [Ringkasan 1-2 kalimat]\n"
            "- Karakteristik Utama: [3-5 poin kunci]\n"
            "- Fakta Unik: [1 poin yang jarang diketahui]\n"
            "Catatan: Tanpa daftar sumber/bibliografi dan tanpa disclaimer generik."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Tulis draf artikel blog singkat (~500 kata) tentang {topic} berdasarkan hasil Peneliti Ahli. "
            "Gaya: hangat, engaging, dan mudah dipahami. "
            "Buat judul yang kuat dengan bentuk '# Judul Artikel'. "
            "Paragraf pembuka harus punya hook (pertanyaan retoris, analogi, atau gambaran singkat). "
            "Gunakan subjudul (##) bila perlu, dan transisi antar paragraf yang halus. "
            "Gunakan variasi panjang kalimat, hindari pengulangan frasa. "
            "Akhiri dengan ringkasan dan dorongan singkat agar pembaca mengambil tindakan/merenung. "
            "JANGAN menambahkan daftar sumber/bibliografi, dan jangan menyebut bahwa Anda adalah model AI."
        ),
        expected_output=(
            "# [Judul Menarik]\n"
            "\n"
            "Paragraf pembuka dengan hook yang memikat.\n"
            "\n"
            "## Subjudul relevan\n"
            "Isi yang menjabarkan poin-poin dari Peneliti (gunakan bahasa alami, tidak kaku).\n"
            "\n"
            "## Subjudul penutup\n"
            "Kesimpulan ringkas + ajakan halus.\n"
            "\n"
            "Catatan: Tanpa daftar sumber/bibliografi, tanpa disclaimer AI."
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
        result = await concurrency.run_in_threadpool(
            content_crew.kickoff, inputs={'topic': topic}
        )
        return {"result": str(result)}
    except RateLimitError:
        # Jika rate limit, tunggu dan coba sekali lagi
        print("Rate limit tercapai. Menunggu 7 detik sebelum mencoba lagi...")
        time.sleep(7)
        result = await concurrency.run_in_threadpool(
            content_crew.kickoff, inputs={'topic': topic}
        )
        return {"result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Selamat datang di CrewAI Content Generation API!"}
