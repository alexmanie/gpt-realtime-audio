"""Speech-to-text conversation with automatic Azure OpenAI replies."""

from __future__ import annotations

import io
import json
import logging
import math
import os
import threading
import wave
from collections.abc import Iterator
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import azure.cognitiveservices.speech as speechsdk
import gradio as gr
import numpy as np
from azure.identity import ChainedTokenCredential, AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from scipy.io import wavfile
from scipy.signal import resample_poly

load_dotenv()

logger = logging.getLogger(__name__)
STT_SAMPLE_RATE = 16000
COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2025-04-01-preview"
)
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT", "").strip()
AZURE_SPEECH_RESOURCE_ID = os.getenv("AZURE_SPEECH_RESOURCE_ID", "").strip()

CONTAINER_TTS_SERAPHINA_ENDPOINT = os.getenv("CONTAINER_TTS_SERAPHINA_ENDPOINT", "").strip()
CONTAINER_TTS_ANDREW_ENDPOINT = os.getenv("CONTAINER_TTS_ANDREW_ENDPOINT", "").strip()
CONTAINER_STT_EN_ENDPOINT = os.getenv("CONTAINER_STT_EN_ENDPOINT", "").strip()
CONTAINER_STT_SL_ENDPOINT = os.getenv("CONTAINER_STT_SL_ENDPOINT", "").strip()

LANGUAGES = {
    "Slovenian (Slovenia)": "sl-SI",
    "English (US)": "en-US",
}
LANGUAGE_NAMES = {
    "sl-SI": "Slovenian",
    "en-US": "English",
}
AVAILABLE_VOICES = [
    "de-DE-SeraphinaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
]
DEFAULT_VOICE = AVAILABLE_VOICES[0]

SYSTEM_INSTRUCTIONS_PATH = Path(__file__).with_name("SYSTEM_INSTRUCTIONS.md")
try:
    SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    SYSTEM_INSTRUCTIONS = (
        "You are a helpful, friendly voice assistant. "
        "Keep responses concise and conversational."
    )


def _speech_config() -> speechsdk.SpeechConfig:
    if not AZURE_SPEECH_ENDPOINT and not AZURE_SPEECH_REGION:
        raise gr.Error("Set AZURE_SPEECH_ENDPOINT or AZURE_SPEECH_REGION for Speech.")

    # location = (
    #     {"endpoint": AZURE_SPEECH_ENDPOINT}
    #     if AZURE_SPEECH_ENDPOINT
    #     else {"region": AZURE_SPEECH_REGION}
    # )
    # if AZURE_SPEECH_KEY:
    #     return speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, **location)

    if ".api.cognitive." in (urlparse(AZURE_SPEECH_ENDPOINT).hostname or ""):
        raise gr.Error(
            "The configured Speech endpoint is a regional API URL, which rejects "
            "credential-based Entra authentication. Set AZURE_SPEECH_ENDPOINT to "
            "your resource-specific custom endpoint "
            "(https://<resource-name>.cognitiveservices.azure.com/), "
            "or set AZURE_SPEECH_KEY for the regional endpoint."
        )
    if not AZURE_SPEECH_ENDPOINT and not AZURE_SPEECH_RESOURCE_ID:
        raise gr.Error(
            "Set AZURE_SPEECH_ENDPOINT for Entra authentication, or set "
            "AZURE_SPEECH_REGION and AZURE_SPEECH_RESOURCE_ID."
        )

    credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(),
    )

    return speechsdk.SpeechConfig(
            token_credential=credential, endpoint=AZURE_SPEECH_ENDPOINT
        )

def _speech_config_container_en_stt() -> speechsdk.SpeechConfig:
    if not CONTAINER_STT_EN_ENDPOINT:
        raise gr.Error("Set CONTAINER_STT_EN_ENDPOINT for Speech-to-text for English on container.")

    return speechsdk.SpeechConfig(
            host=CONTAINER_STT_EN_ENDPOINT
        )


def _speech_config_container_sl_stt() -> speechsdk.SpeechConfig:
    if not CONTAINER_STT_SL_ENDPOINT:
        raise gr.Error("Set CONTAINER_STT_SL_ENDPOINT for Speech-to-text for Slovenian on container.")

    return speechsdk.SpeechConfig(
            host=CONTAINER_STT_SL_ENDPOINT
        )


def _speech_config_container_tts(voice: str) -> speechsdk.SpeechConfig:
    if voice == "de-DE-SeraphinaMultilingualNeural":
        if not CONTAINER_TTS_SERAPHINA_ENDPOINT:
            raise gr.Error("Set CONTAINER_TTS_SERAPHINA_ENDPOINT for Text-to-speech on container.")
        return speechsdk.SpeechConfig(host=CONTAINER_TTS_SERAPHINA_ENDPOINT)
    elif voice == "en-US-AndrewMultilingualNeural":
        if not CONTAINER_TTS_ANDREW_ENDPOINT:
            raise gr.Error("Set CONTAINER_TTS_ANDREW_ENDPOINT for Text-to-speech on container.")
        return speechsdk.SpeechConfig(host=CONTAINER_TTS_ANDREW_ENDPOINT)
    else:
        raise gr.Error(f"Unsupported voice for container: {voice}")



def transcribe_audio(audio_path: str, language: str, use_container: bool = False) -> str:
    if not audio_path or not Path(audio_path).is_file():
        raise gr.Error("Record or upload an audio clip before sending.")

    pcm, duration, audio_details = _prepare_stt_audio(audio_path)
    
    if use_container:
        if language == "en-US":
            speech_config = _speech_config_container_en_stt()
        elif language == "sl-SI":
            speech_config = _speech_config_container_sl_stt()
        else:
            raise gr.Error(f"Unsupported language for container: {language}")
    else:
        speech_config = _speech_config()
    
    speech_config.speech_recognition_language = language
    
    audio_stream = speechsdk.audio.PushAudioInputStream(
        stream_format=speechsdk.audio.AudioStreamFormat(
            samples_per_second=STT_SAMPLE_RATE, bits_per_sample=16, channels=1
        )
    )
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=speechsdk.AudioConfig(stream=audio_stream),
    )
    finished = threading.Event()
    segments: list[str] = []
    errors: list[str] = []
    no_matches: list[str] = []
    cancellations: list[str] = []

    def recognized(event) -> None:
        if event.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = event.result.text.strip()
            if text:
                segments.append(text)
        elif event.result.reason == speechsdk.ResultReason.NoMatch:
            details = event.result.no_match_details
            reason = str(details.reason) if details is not None else "NoMatch"
            no_matches.append(reason)
            logger.warning(
                "Speech NoMatch: %s; %s; language=%s", reason, audio_details, language
            )

    def canceled(event) -> None:
        cancellations.append(str(event.reason))
        if event.reason == speechsdk.CancellationReason.Error:
            errors.append(event.error_details or str(event.reason))
        finished.set()

    recognizer.recognized.connect(recognized)
    recognizer.canceled.connect(canceled)
    recognizer.session_stopped.connect(lambda event: finished.set())
    try:
        recognizer.start_continuous_recognition_async().get()
        try:
            audio_stream.write(pcm)
            audio_stream.close()
            if not finished.wait(timeout=duration + 30):
                raise gr.Error("Speech recognition timed out. Please try again.")
        finally:
            recognizer.stop_continuous_recognition_async().get()
    finally:
        audio_stream.close()
    if errors:
        raise gr.Error(f"Speech recognition failed: {errors[0]}")
    if not segments:
        reason = "; ".join(no_matches or cancellations) or (
            "The Speech session stopped without recognition results; "
            "check the endpoint and authentication configuration"
        )
        raise gr.Error(
            f"No speech was recognized: {reason}. {audio_details}. "
            f"Recognition language: {language}. Play back the recording to check "
            "that speech is audible, and select the language you spoke."
        )
    return " ".join(segments)


def _prepare_stt_audio(audio_path: str) -> tuple[bytes, float, str]:
    try:
        sample_rate, samples = wavfile.read(audio_path)
    except (ValueError, OSError) as exc:
        raise gr.Error(f"Cannot read the recording as WAV audio: {exc}") from exc
    if sample_rate <= 0 or samples.size == 0:
        raise gr.Error("The recording is empty. Record a message before sending.")
    channels = samples.shape[1] if samples.ndim == 2 else 1
    source_dtype = str(samples.dtype)
    if samples.dtype == np.uint8:
        samples = (samples.astype(np.float64) - 128) / 128
    elif np.issubdtype(samples.dtype, np.signedinteger):
        samples = samples.astype(np.float64) / (2 ** (samples.dtype.itemsize * 8 - 1))
    elif np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float64)
    else:
        raise gr.Error(f"Unsupported recording sample type: {source_dtype}.")
    if not np.isfinite(samples).all():
        raise gr.Error("The recording contains invalid audio samples. Record it again.")
    if channels > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples.reshape(-1)
    if sample_rate != STT_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, STT_SAMPLE_RATE)
        samples = resample_poly(
            samples, STT_SAMPLE_RATE // divisor, sample_rate // divisor
        )
    pcm = np.rint(np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2")
    duration = pcm.size / STT_SAMPLE_RATE
    peak = float(np.max(np.abs(pcm.astype(np.float64)))) / 32768
    rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768) ** 2)))
    rms_db = f"{20 * math.log10(rms):.1f} dBFS" if rms else "-inf dBFS"
    details = (
        f"Recording: {duration:.2f}s, {sample_rate} Hz, {channels} channel(s), "
        f"{source_dtype}; sent as 16000 Hz mono PCM16; "
        f"peak {peak:.4f}, RMS {rms_db}"
    )
    logger.info("STT audio: %s", details)
    if not pcm.any():
        raise gr.Error(
            f"The recording contains only silence. {details}. "
            "Check microphone permissions and the selected input device."
        )
    return pcm.tobytes(), duration, details


def send_to_stt(
    audio_path: str | None,
    language: str,
    history: list[dict[str, str]] | None,
    use_container: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    transcript = transcribe_audio(audio_path or "", language, use_container)
    updated_history = [
        *(history or []),
        {"role": "user", "content": transcript},
    ]
    return updated_history, updated_history


def generate_reply(history: list[dict[str, str]], language: str) -> str:
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_CHAT_DEPLOYMENT:
        raise gr.Error(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_CHAT_DEPLOYMENT to a "
            "chat-capable Azure OpenAI deployment."
        )

    credential = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())
    client: AzureOpenAI | None = None
    try:
        token_provider = get_bearer_token_provider(
            credential,
            COGNITIVE_SERVICES_SCOPE,
        )
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_INSTRUCTIONS}\nRespond in the language "
                f"{LANGUAGE_NAMES.get(language, language)} ({language}).",
            },
            *history,
        ]
        result = client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=messages,
        )
    finally:
        if client is not None:
            client.close()
        credential.close()

    reply = result.choices[0].message.content
    if not reply or not reply.strip():
        raise gr.Error("The chat deployment returned an empty response.")
    return reply.strip()


def respond(
    audio_path: str | None,
    language: str,
    history: list[dict[str, str]] | None,
    use_container: bool = False,
) -> Iterator[
    tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str] | None]
]:
    dialog, state = send_to_stt(audio_path, language, history, use_container)
    yield dialog, state, gr.skip()
    reply = generate_reply(state, language)
    updated_history = [*state, {"role": "assistant", "content": reply}]
    yield updated_history, updated_history, None


def synthesize_speech(
    text: str, voice: str, use_container: bool = False
) -> tuple[int, np.ndarray]:
    
    if use_container:
        speech_config = _speech_config_container_tts(voice)
    else:
        speech_config = _speech_config()

    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        message = details.error_details or str(details.reason)
        raise gr.Error(f"Speech synthesis failed: {message}")
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise gr.Error("Speech synthesis did not return audio.")

    with wave.open(io.BytesIO(result.audio_data), "rb") as audio_file:
        if audio_file.getnchannels() != 1 or audio_file.getsampwidth() != 2:
            raise gr.Error("Speech synthesis returned an unsupported audio format.")
        sample_rate = audio_file.getframerate()
        audio = np.frombuffer(
            audio_file.readframes(audio_file.getnframes()),
            dtype=np.int16,
        ).copy()
    return sample_rate, audio


def speak_last_reply(
    history: list[dict[str, str]] | None, voice: str, use_container: bool = False
) -> tuple[int, np.ndarray]:
    for message in reversed(history or []):
        if message["role"] == "assistant" and message["content"].strip():
            return synthesize_speech(message["content"], voice, use_container)
    raise gr.Error("There is no assistant reply to convert to speech yet.")


def _container_readiness(endpoint: str) -> tuple[bool, str]:
    if not endpoint:
        return False, "Endpoint not configured"
    try:
        with urlopen(f"{endpoint.rstrip('/')}/ready", timeout=3) as response:
            payload = json.load(response)
    except (OSError, ValueError) as exc:
        return False, f"Readiness check failed: {exc}"
    if not isinstance(payload, dict):
        return False, "Invalid readiness response: expected a JSON object"
    ready = payload.get("ready") == "ready"
    message = str(payload.get("message") or (
        "Service is ready" if ready else 'Response does not contain "ready": "ready"'
    ))
    return ready, message


def check_container_status() -> str:
    indicators = []
    for label, endpoint in (
        ("STT English", CONTAINER_STT_EN_ENDPOINT),
        ("STT Slovenian", CONTAINER_STT_SL_ENDPOINT),
        ("TTS Seraphina", CONTAINER_TTS_SERAPHINA_ENDPOINT),
        ("TTS Andrew", CONTAINER_TTS_ANDREW_ENDPOINT),
    ):
        ready, message = _container_readiness(endpoint)
        color = "#16a34a" if ready else "#dc2626"
        background = "#f0fdf4" if ready else "#fef2f2"
        border = "#bbf7d0" if ready else "#fecaca"
        status = "Ready" if ready else "Not ready"
        indicators.append(
            f'<div title="{escape(message, quote=True)}" style="display:flex;'
            f'align-items:center;gap:8px;padding:8px 12px;border:1px solid {border};'
            f'border-radius:6px;background:{background};color:{color};'
            'font-size:12px;line-height:20px;white-space:nowrap">'
            f'<span aria-hidden="true" style="display:inline-block;width:6px;'
            f'height:6px;flex-shrink:0;border-radius:50%;background:{color}"></span>'
            f'<span>{escape(label)}: {status}</span></div>'
        )
    return (
        '<div role="status" aria-label="Speech container status" '
        'style="display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px">'
        + "".join(indicators) + "</div>"
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Speech and OpenAI conversation") as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=420):
                gr.Markdown("# Speech and OpenAI Conversation")
            with gr.Column(scale=2, min_width=320):
                container_status = gr.HTML(
                    '<div role="status" style="text-align:right">'
                    'Checking Speech containers...</div>'
                )
        status_timer = gr.Timer(30)
        gr.Markdown(
            f"""
            Record your microphone, stop recording, then click **Send**. 
            The transcription appears as your message in the conversation.  
            Azure OpenAI replies automatically using deployment
            **{AZURE_OPENAI_CHAT_DEPLOYMENT or "(not configured)"}**.  
            Use **Speak last reply** below the conversation to generate audio.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                use_container = gr.Checkbox(
                    label="Use Speech containers",
                    value=True,
                    interactive=False,
                    info=f"""
                    Off: online Azure Speech.  
                    On: configured Speech containers.""",
                )
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    format="wav",
                    label="Your message",
                )
                language_dropdown = gr.Dropdown(
                    choices=[(name, locale) for name, locale in LANGUAGES.items()],
                    value="sl-SI",
                    label="Spoken language",
                )
                send_button = gr.Button("Send", variant="primary")
                clear_button = gr.Button("Clear conversation")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=420,
                )
                conversation_state = gr.State([])
                voice_dropdown = gr.Dropdown(
                    choices=AVAILABLE_VOICES,
                    value=DEFAULT_VOICE,
                    label="Reply voice",
                )
                speak_button = gr.Button("Speak last reply")
                audio_output = gr.Audio(
                    label="Spoken reply", interactive=False, format="wav", autoplay=True
                )

        demo.load(
            fn=check_container_status,
            outputs=container_status,
            queue=False,
            show_progress="hidden",
        )
        status_timer.tick(
            fn=check_container_status,
            outputs=container_status,
            queue=False,
            show_progress="hidden",
        )
        send_button.click(
            fn=respond,
            inputs=[audio_input, language_dropdown, conversation_state, use_container],
            outputs=[chatbot, conversation_state, audio_input],
            concurrency_id="conversation",
        ).success(
            fn=lambda: None,
            outputs=audio_output,
            concurrency_id="conversation",
        )
        speak_button.click(
            fn=speak_last_reply,
            inputs=[conversation_state, voice_dropdown, use_container],
            outputs=audio_output,
            concurrency_id="conversation",
        )
        clear_button.click(
            fn=lambda: ([], [], None, None),
            outputs=[chatbot, conversation_state, audio_input, audio_output],
            concurrency_id="conversation",
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        show_error=True,
    )
