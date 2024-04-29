from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import base64
import requests
import os

# Load .env file
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request, image: UploadFile = File(...), claims: str = Form(...)):
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    api_key = os.getenv('OPENAI_API_KEY')

    # Error handling if API key is not set
    if not api_key:
        return JSONResponse(content={"message": "OpenAI API key is not set."}, status_code=500)

    headers = {"Authorization": f"Bearer {api_key}"}
    system_prompt = """  ConsumeWise analyzes product labels to validate claims and identify discrepancies. It reports findings as follows:
    1. An initial verdict stating whether the claim is '100% right' or if something 'fishy' was found.
    2. A detailed explanation of any incongruities or misleading elements.
    The GPT integrates knowledge from various articles on healthwashing to dissect how products might exploit regulatory loopholes or present misleading health claims. It also evaluates the realistic consumption needed to achieve the claimed nutritional benefits, considering potential health impacts.   """

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": claims},
            {"role": "assistant", "content": f"data:image/jpeg;base64,{base64_image}"}
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Check response status and handle any errors
    if response.status_code != 200:
        return JSONResponse(content={"message": "Failed to get a valid response from OpenAI API."}, status_code=response.status_code)

    try:
        response_text = response.json()['choices'][0]['message']['content']
    except KeyError as e:
        response_text = f"An error occurred when parsing the response: {e}"

    return templates.TemplateResponse("index.html", {"request": request, "response": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
