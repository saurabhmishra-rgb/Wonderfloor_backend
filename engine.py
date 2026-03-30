import os
import io
import base64
from typing import Optional

import httpx
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://wonderfloor.onrender.com"
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

client = OpenAI(api_key=API_KEY)


def prepare_image(image_bytes: bytes, filename: str, max_dim: int = 1536):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        w, h = img.size
        scale = min(max_dim / w, max_dim / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if scale < 1.0:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        stream = io.BytesIO()
        img.save(stream, format="PNG")
        stream.seek(0)

        safe_name = os.path.splitext(filename or "image")[0] + ".png"
        return (safe_name, stream, "image/png")
    except Exception as e:
        raise ValueError(f"Image processing failed: {e}")


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
        f"Flooring reference name: {floor_style}.",
    ]

    if instructions and instructions.strip():
        prompt_parts.append(f"Style detail: {instructions.strip()}")

    full_prompt = " ".join(prompt_parts)

    try:
        print(f"🚀 Model: {MODEL}")
        print(f"🚀 Floor reference file: {floorImage.filename}")
        print(f"🚀 Prompt: {full_prompt}")

        room_bytes = await roomImage.read()
        floor_bytes = await floorImage.read()

        if not room_bytes or not floor_bytes:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "One or both uploaded files are empty."},
            )

        room_file = prepare_image(
            room_bytes,
            roomImage.filename or "room.png",
            max_dim=1536,
        )
        floor_file = prepare_image(
            floor_bytes,
            floorImage.filename or "floor.png",
            max_dim=1024,
        )

        result = client.images.edit(
            model=MODEL,
            image=[room_file, floor_file],
            prompt=full_prompt,
            n=1,
            size="1536x1024",
            quality="medium",
        )

        usage = result.usage
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        input_details = getattr(usage, "input_tokens_details", None)
        text_tokens = getattr(input_details, "text_tokens", 0) if input_details else 0
        image_tokens = getattr(input_details, "image_tokens", 0) if input_details else 0

        print("===== OPENAI USAGE =====")
        print("input_tokens:", input_tokens)
        print("output_tokens:", output_tokens)
        print("total_tokens:", total_tokens)
        print("text_tokens:", text_tokens)
        print("image_tokens:", image_tokens)
        print("========================")

        image_base64 = result.data[0].b64_json

        return {
            "success": True,
            "imageDataUrl": f"data:image/png;base64,{image_base64}",
            "usedModel": MODEL,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
            },
        }

    except ValueError as ve:
        print(f"❌ Processing error: {ve}")
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": str(ve)},
        )
    except Exception as e:
        print(f"❌ API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


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
        return JSONResponse(
            status_code=502,
            content={"success": False, "error": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Starting on http://127.0.0.1:{PORT} | Model: {MODEL}")
    uvicorn.run("engine:app", host="127.0.0.1", port=PORT, reload=True)
