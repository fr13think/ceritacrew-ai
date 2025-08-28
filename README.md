Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1

python -m uvicorn main:app --reload

pip install --force-reinstall "uvicorn[standard]" fastapi crewai python-dotenv langchain-groq