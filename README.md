Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
.\venv\Scripts\Activate

python -m pip install --upgrade pip

pip install -r requirement.txt

python -m uvicorn main:app --reload

pip install langchain-community
pip install -U langchain-litellm

python -m http.server 5500

Tes cepat:
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{"topic":"tes"}"

pip install -U fastapi uvicorn python-dotenv crewai litellm openai


Tips cepat untuk hasil lebih “menarik”:
Coba model lebih besar: set NIM_MODEL=openai/meta/llama-3.3-70b-instruct di .env (akan lebih kreatif, tapi lebih lambat).
Atur writer lebih kreatif: temperature 1.0, top_p 0.95–0.98, presence_penalty 0.7.
Tambah “gaya bahasa” spesifik (mis. analogi, storytelling singkat, kata ganti “Anda”, target pembaca, nada santai).
Pastikan max_tokens cukup (≥ 700) agar struktur lengkap bisa tercetak.