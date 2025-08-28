Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
.\venv\Scripts\Activate

python -m pip install --upgrade pip

pip install -r requirement.txt

python -m uvicorn main:app --reload

pip install langchain-community
pip install -U langchain-litellm

