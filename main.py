from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

recipe_processor = None

@app.on_event("startup")
async def startup_event():
    global recipe_processor
    try:
        print("Initializing RecipeProcessor...")
        from app import RecipeProcessor
        recipe_processor = RecipeProcessor()
        print("RecipeProcessor initialized successfully!")
    except Exception as e:
        print("STARTUP ERROR - RecipeProcessor failed to initialize:")
        print(traceback.format_exc())

class MessageRequest(BaseModel):
    message: str

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/echo")
def analyze(request: MessageRequest):
    if recipe_processor is None:
        return {"error": "Model failed to initialize. Check server logs."}
    result = recipe_processor.analyze_recipe(request.message)
    print("Analysis result: ", result.get("top_candidate", "error"))
    return result