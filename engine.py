import os
import io
import base64
import asyncio
import time
from typing import Optional

import httpx
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://wonderfloor1.onrender.com/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORT = int(os.getenv("PORT", "8000"))
MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("⚠️ WARNING: Missing OPENAI_API_KEY.")

# Initialize client with max_retries=0 to prevent 2-minute hangs on API errors
client = AsyncOpenAI(api_key=API_KEY, max_retries=0)


def prepare_image(image_bytes: bytes, filename: str, max_dim: int = 1024):
    """Synchronous CPU-bound function to process the image."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        w, h = img.size
        scale = min(max_dim / w, max_dim / h, 1.0)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        stream = io.BytesIO()
        img.save(stream, format="PNG")
        stream.seek(0)

        safe_name = os.path.splitext(filename or "image")[0] + ".png"
        return (safe_name, stream, "image/png")
    except Exception as e:
        raise ValueError(f"Image processing failed: {e}")


@app.get("/")
def read_root():
    return {"message": "Wonderfloor API is running smoothly!"}


@app.get("/api/health")
def health_check():
    return {
        "ok": True,
        "status": "online",
        "port": PORT,
        "model": MODEL,
    }


@app.post("/api/replace-floor")
async def replace_floor(
    roomImage: UploadFile = File(...),
    floorImage: UploadFile = File(...),
    instructions: Optional[str] = Form(""),
):
    total_start_time = time.time()
    
    if not roomImage or not floorImage:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Upload both images."},
        )

    floor_style = (
        os.path.splitext(floorImage.filename or "floor")[0]
        .replace("_", " ")
        .replace("-", " ")
    )

    prompt_parts = [
        "Edit the first image only.",
        "Replace only the visible floor area in the first image.",
        "Use the second image as the flooring material reference.",
        "Match the second image as closely as possible in color, pattern, texture, finish, and scale.",
        "Keep all walls, furniture, lighting, shadows, and room structure exactly the same.",
        "The result must look photorealistic and only the floor surface should change.",
        "The floor must appear as a single continuous seamless surface with no visible plank lines, joints, seams, or tile divisions. Do not show individual planks or segmented boards.",
        f"Flooring reference name: {floor_style}.",
    ]

    if instructions and instructions.strip():
        prompt_parts.append(f"Style detail: {instructions.strip()}")

    full_prompt = " ".join(prompt_parts)

    try:
        print(f"\n--- NEW REQUEST STARTED ---")
        print(f"🚀 Model: {MODEL}")
        
        # 1. Read files
        t0 = time.time()
        room_bytes, floor_bytes = await asyncio.gather(
            roomImage.read(),
            floorImage.read()
        )
        print(f"⏱️ File Reading took: {time.time() - t0:.2f} seconds")

        if not room_bytes or not floor_bytes:
            return JSONResponse(status_code=400, content={"success": False, "error": "Empty files."})

        # 2. Process images (Reduced room image max_dim to 1024 to speed up AI generation)
        t1 = time.time()
        room_file, floor_file = await asyncio.gather(
            asyncio.to_thread(prepare_image, room_bytes, roomImage.filename or "room.png", 1024),
            asyncio.to_thread(prepare_image, floor_bytes, floorImage.filename or "floor.png", 1024)
        )
        print(f"⏱️ Image Processing (Pillow) took: {time.time() - t1:.2f} seconds")

        # 3. OpenAI API Call (Requested standard 1024x1024 and removed 'medium' quality for speed)
        print("⏳ Sending request to OpenAI API...")
        t2 = time.time()
        result = await client.images.edit(
            model=MODEL,
            image=[room_file, floor_file],
            prompt=full_prompt,
            n=1,
            size="1024x1024",
        )
        print(f"⏱️ OpenAI API Call took: {time.time() - t2:.2f} seconds")

        image_base64 = result.data[0].b64_json
        
        print(f"✅ Total Request Time: {time.time() - total_start_time:.2f} seconds")
        print("---------------------------\n")

        return {
            "success": True,
            "imageDataUrl": f"data:image/png;base64,{image_base64}",
            "usedModel": MODEL,
        }

    except ValueError as ve:
        print(f"❌ Processing error: {ve}")
        return JSONResponse(status_code=422, content={"success": False, "error": str(ve)})
    except Exception as e:
        print(f"❌ API error: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/proxy-tile")
async def proxy_tile(url: str):
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            return {
                "success": True,
                "contentType": resp.headers.get("content-type", "image/jpeg"),
                "data": base64.b64encode(resp.content).decode(),
            }
    except Exception as e:
        return JSONResponse(status_code=502, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting on http://127.0.0.1:{PORT} | Model: {MODEL}")
    uvicorn.run("engine:app", host="127.0.0.1", port=PORT, reload=True)
