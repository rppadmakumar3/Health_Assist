from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import subprocess

app = FastAPI()


@app.post("/train-model")
async def train_model():
    try:
        # Define the training command
        cmd = "python src/medical_diagnosis_initial_training.py --datadir ./data/chest_xray"

        # Execute the command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout

        return {"status": "success", "output": output}
    except Exception as e:
        return {"status": "error", "message": str(e)}
