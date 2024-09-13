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
from scipy.signal import resample

from pydantic import BaseModel, Field, constr, model_validator
from typing import Optional, Literal

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


class InferenceRequest(BaseModel):
    query: str
    role: Optional[str] = "中文男"
    instruct: Optional[str] = (
        "Ultraman Tiga, the Giant of Light, is a brave and determined guardian of Earth. He is full of a sense of justice and passion."
    )
    bit_depth: Optional[Literal[8, 16, 24, 32]] = None  # 可选参数，支持的位深
    # sample_rate: Optional[Literal[16000, 22050, 44100]] = None  # 可选参数，支持的采样率
    # TODO: 格式待后续实现
    # format: Optional[Literal["acc", "wav", "ogg", "mp3"]] = None  # 可选参数，支持的格式

    # @model_validator(mode="before")
    # def check_bitDepth_and_sampleRate(cls, values):
    #     bit_depth = values.get("bit_depth")
    #     sample_rate = values.get("sample_rate")
    #
    #     if (bit_depth is None) != (sample_rate is None):
    #         raise ValueError(
    #             "bit_depth and sample_rate must both be provided or both be omitted."
    #         )
    #     return values


def generate_data(model_output, bit_depth=None):
    for i in model_output:
        samples = i["tts_speech"].numpy()

        # Convert to desired bit depth
        if bit_depth == 8:
            samples = ((samples + 1.0) * 127.5).astype(np.uint8)  # pyright: ignore
        elif bit_depth == 16:
            samples = (samples * 32767).astype(np.int16)  # pyright: ignore
        elif bit_depth == 24:
            samples = (samples * 8388607).astype(np.int32)  # pyright: ignore
            samples_bytes = samples.astype(np.int32).tobytes()
            # Extract 3 bytes per sample
            samples = (
                np.frombuffer(samples_bytes, dtype=np.uint8)
                .reshape(-1, 4)[:, :3]
                .flatten()
            )
        elif bit_depth == 32:
            samples = (samples * 2147483647).astype(np.int32)  # pyright: ignore

        tts_audio = samples.tobytes()  # pyright: ignore
        yield tts_audio


@app.get("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form()):
    model_output = cosyvoice.inference_sft(tts_text, spk_id)
    return StreamingResponse(generate_data(model_output))


@app.post("/inference/stream")
async def stream(request: InferenceRequest):
    tts_text = request.query
    role = request.role
    instruct = request.instruct
    bit_depth = request.bit_depth

    if not tts_text:
        raise HTTPException(
            status_code=400, detail="Query parameter 'query' is required"
        )

    if not instruct:
        raise HTTPException(
            status_code=400, detail="Query parameter 'instruct' is required"
        )

    model_output = cosyvoice.inference_instruct(tts_text, role, instruct, stream=True)
    return StreamingResponse(generate_data(model_output, bit_depth))


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
