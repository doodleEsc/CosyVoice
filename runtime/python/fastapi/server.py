# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from fastapi import FastAPI, HTTPException, UploadFile, Form, File, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../..".format(ROOT_DIR))
sys.path.append("{}/../../../third_party/Matcha-TTS".format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_data(model_output):
    for i in model_output:
        # tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        tts_audio = i["tts_speech"].numpy().tobytes()
        yield tts_audio


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.post("/inference_instruct/stream")
async def instruct_stream(request: Request):
    question_data = await request.json()
    tts_text = question_data.get("query")
    role = question_data.get("role")
    instruct = question_data.get("instruct")
    if not tts_text:
        raise HTTPException(
            status_code=400, detail="Query parameter 'query' is required"
        )

    if not instruct:
        instruct = "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness."

    model_output = cosyvoice.inference_instruct(tts_text, role, instruct, stream=True)
    return StreamingResponse(generate_data(model_output))


@app.post("/inference/stream")
async def stream(request: Request):
    question_data = await request.json()
    tts_text = question_data.get("query")
    role = question_data.get("role")
    if not tts_text:
        raise HTTPException(
            status_code=400, detail="Query parameter 'query' is required"
        )

    model_output = cosyvoice.inference_sft(tts_text, role, stream=True)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_zero_shot(
        tts_text, prompt_text, prompt_speech_16k
    )
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_cross_lingual")
async def inference_cross_lingual(
    tts_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct")
async def inference_instruct(
    tts_text: str = Form(), spk_id: str = Form(), instruct_text: str = Form()
):
    model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
    return StreamingResponse(generate_data(model_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="iic/CosyVoice-300M",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir, load_onnx=False)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
