"""
Speech-to-speech demo using Azure AI Foundry gpt-realtime model.

Runs a Gradio UI with WebRTC (via FastRTC) for bidirectional low-latency audio.
No extra Flask layer needed - Gradio handles UI + WebSocket bridge to Foundry.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import os

import gradio as gr
import numpy as np
from azure.identity.aio import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential, get_bearer_token_provider
from dotenv import load_dotenv
from fastrtc import (
    AsyncStreamHandler,
    Stream,
    get_cloudflare_turn_credentials_async,
    get_cloudflare_turn_credentials,
    wait_for_item,
)
from openai import AsyncAzureOpenAI

load_dotenv()

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-realtime-1.5")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

# Load system instructions from markdown file
def _load_system_instructions() -> str:
    try:
        with open("SYSTEM_INSTRUCTIONS.md", "r") as f:
            content = f.read()
            # # Remove markdown title if present
            # lines = content.strip().split("\n")
            # if lines and lines[0].startswith("#"):
            #     return "\n".join(lines[1:]).strip()
            return content.strip()
    except FileNotFoundError:
        return "You are a helpful, friendly voice assistant. Keep responses concise and conversational."

SYSTEM_INSTRUCTIONS = _load_system_instructions()

SAMPLE_RATE = 24000  # gpt-realtime uses 24kHz PCM16
AVAILABLE_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
]
DEFAULT_VOICE = "marin"
SELECTED_VOICE = DEFAULT_VOICE

# Pointer-events fix: the FastRTC waveContainer is absolutely positioned and
# covers the whole component area. Setting pointer-events:none on it lets
# clicks pass through to Gradio widgets rendered below it.
UI_CSS = """
gradio-webrtc-waveContainer,
.gradio-webrtc-waveContainer {
    pointer-events: none !important;
}

#voice-controls {
    position: relative;
    z-index: 20;
    margin-top: 0;
    margin-bottom: 0.75rem;
}
"""


def get_selected_voice() -> str:
    return SELECTED_VOICE


def set_selected_voice(voice: str) -> str:
    global SELECTED_VOICE
    if voice in AVAILABLE_VOICES:
        SELECTED_VOICE = voice
    return SELECTED_VOICE


class RealtimeHandler(AsyncStreamHandler):
    """Bridges FastRTC audio frames to Azure Foundry gpt-realtime WebSocket."""

    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            input_sample_rate=SAMPLE_RATE,
        )
        self.output_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.connection = None
        self.client: AsyncAzureOpenAI | None = None
        self.credential: ChainedTokenCredential | None = None

    def copy(self) -> "RealtimeHandler":
        return RealtimeHandler()

    async def start_up(self) -> None:
        if not AZURE_ENDPOINT:
            raise RuntimeError(
                "Missing AZURE_OPENAI_ENDPOINT. Copy .env.example to .env and set it."
            )

        # Entra ID auth: try DefaultAzureCredential first (env vars, Managed Identity,
        # VS Code, etc.), then fall back to AzureCliCredential for local `az login`.
        # The signed-in principal needs the 'Cognitive Services OpenAI User' role
        # on the Foundry resource.
        self.credential = ChainedTokenCredential(
            DefaultAzureCredential(),
            AzureCliCredential(),
        )
        token_provider = get_bearer_token_provider(
            self.credential, COGNITIVE_SERVICES_SCOPE
        )

        self.client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=AZURE_API_VERSION,
        )
        selected_voice = get_selected_voice()

        async with self.client.beta.realtime.connect(model=AZURE_DEPLOYMENT) as conn:
            self.connection = conn
            await conn.session.update(
                session={
                    "modalities": ["audio", "text"],
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "voice": selected_voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                }
            )

            async for event in conn:
                etype = event.type
                if etype == "response.audio.delta":
                    pcm = base64.b64decode(event.delta)
                    audio = np.frombuffer(pcm, dtype=np.int16).reshape(1, -1)
                    await self.output_queue.put((SAMPLE_RATE, audio))
                elif etype == "response.audio_transcript.delta":
                    # Optional: could surface transcript to UI
                    # Get transcript text from the delta
                    transcript_delta = event.delta
                    pass
                elif etype == "error":
                    print(f"[realtime error] {event}")

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if self.connection is None:
            return
        _sr, array = frame
        array = array.squeeze()
        if array.dtype != np.int16:
            array = array.astype(np.int16)
        audio_b64 = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.input_audio_buffer.append(audio=audio_b64)

    async def emit(self) -> tuple[int, np.ndarray] | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception:
                pass
            self.connection = None
        if self.credential is not None:
            try:
                await self.credential.close()
            except Exception:
                pass
            self.credential = None


def build_ui() -> gr.Blocks:
    # WebRTC needs a TURN server when the browser and server are on different
    # networks (i.e. anywhere except localhost). Azure Container Apps ingress
    # does not forward the UDP traffic ICE negotiates, so without TURN the
    # browser reports "Connection failed".
    #
    # If CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_KEY_API_TOKEN are set, use
    # Cloudflare's TURN service (free tier: 10 GB/month). Otherwise fall back
    # to STUN-only, which only works on localhost.
    cf_key_id = os.getenv("CLOUDFLARE_TURN_KEY_ID")
    cf_key_token = os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN")

    stream_kwargs: dict = {}
    if cf_key_id and cf_key_token:
        # Server-side creds fetcher (used by FastRTC's /webrtc/offer endpoint
        # when the client connects) and a sync variant used at page load.
        async def _rtc_creds():
            return await get_cloudflare_turn_credentials_async(
                turn_key_id=cf_key_id,
                turn_key_api_token=cf_key_token,
                ttl=3600,
            )

        stream_kwargs["rtc_configuration"] = _rtc_creds
        stream_kwargs["server_rtc_configuration"] = get_cloudflare_turn_credentials(
            turn_key_id=cf_key_id,
            turn_key_api_token=cf_key_token,
            ttl=3600,
        )

    stream = Stream(
        handler=RealtimeHandler(),
        modality="audio",
        mode="send-receive",
        ui_args={
            "title": "",
            "subtitle": "",
            "full_screen": False,
        },
        **stream_kwargs,
    )

    # Gradio 5.x: css goes in Blocks(); Gradio 6.x: css goes in launch().
    # Pass css here only for Gradio 5.x (where launch() doesn't accept it).
    _blocks_kwargs: dict = {"title": "Multilanguage speech-to-speech"}
    if "css" not in inspect.signature(gr.Blocks.launch).parameters:
        _blocks_kwargs["css"] = UI_CSS

    with gr.Blocks(**_blocks_kwargs) as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    f"""
                    # 🎙️ Multi-language Speech-to-Speech Demo
                    Talking to **{AZURE_DEPLOYMENT}** on Azure AI Foundry.

                    Click **Record**, allow microphone access, and start speaking.
                    Server-side VAD will detect when you stop and the model will reply with audio.
                    """
                )

                with gr.Group(elem_id="voice-controls"):
                    voice_dropdown = gr.Dropdown(
                        label="Voice",
                        choices=AVAILABLE_VOICES,
                        value=get_selected_voice(),
                        info="Select the voice used for new realtime sessions.",
                    )
                    voice_status = gr.Markdown(
                        f"Current voice: **{get_selected_voice()}**. Stop/start recording to apply changes."
                    )

                def on_voice_change(voice: str) -> str:
                    selected = set_selected_voice(voice)
                    return (
                        f"Current voice: **{selected}**. "
                        "Stop/start recording to apply changes."
                    )

                voice_dropdown.change(
                    fn=on_voice_change,
                    inputs=voice_dropdown,
                    outputs=voice_status,
                )

                stream.ui.render()

            with gr.Column():
                transcript_box = gr.Textbox(
                    label="Transcript",
                    placeholder="Text received from realtime model goes here.",
                    interactive=False,
                    lines=20,
                )

    return demo


if __name__ == "__main__":
    app = build_ui()
    _launch_kwargs: dict = {
        "server_name": os.getenv("GRADIO_HOST", "127.0.0.1"),
        "server_port": int(os.getenv("GRADIO_PORT", "7860")),
        "show_error": True,
    }
    # Gradio 6+ moved css from Blocks() to launch(); keep compatible with both.
    if "css" in inspect.signature(app.launch).parameters:
        _launch_kwargs["css"] = UI_CSS
    app.launch(**_launch_kwargs)
