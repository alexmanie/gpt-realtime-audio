# Azure Speech Assistant

A Gradio interface using **Azure AI Speech** for speech-to-text (STT) and
on-demand text-to-speech (TTS), with a shared editable text area.

## Architecture

```text
Browser microphone -> Gradio server -> Azure AI Speech STT endpoint
Editable text area <- recognized text <-
Editable text area -> Azure AI Speech TTS endpoint -> Audio player
```

Each recorded turn is uploaded to the Python app as a WAV file. The server
sends it to the configured Speech endpoint, collects all recognized sentences,
and replaces the text area's contents with the transcription.
You can edit the transcription or type your own text before converting it to audio.
There is no conversation history or automatic OpenAI response; Azure OpenAI and
`SYSTEM_INSTRUCTIONS.md` are not used. Text is isolated to the current Gradio
session. Recognition errors are shown in the frontend without changing the
existing text or clearing the recording.
WebRTC and TURN credentials are not required.

Browser recordings are converted to 16 kHz, mono, signed 16-bit PCM before
being sent to Speech. Empty or silent recordings are rejected locally.
If recognition returns no text, the frontend shows Azure's no-match reason,
the selected language, and audio duration/level details. Play the recording
back to verify the microphone captured speech and check the language selector.

## Prerequisites

- Python 3.10+
- FFmpeg installed with both `ffmpeg` and `ffprobe` on `PATH` for MP3 and
  other non-WAV uploads (included in the Docker image)
- An Azure AI Speech resource and its endpoint (or region)
- Azure CLI (`az`) for local Entra ID sign-in, or a managed identity when hosted

## Setup

```powershell
az login
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

- `AZURE_SPEECH_ENDPOINT` to the Speech resource's dedicated endpoint, for example
  `https://your-speech-resource.cognitiveservices.azure.com/`. Do not use an
  Azure OpenAI endpoint. Alternatively, leave it unset and configure
  `AZURE_SPEECH_REGION` to use the regional Speech endpoint.
- For Microsoft Entra ID, use the resource-specific custom endpoint and sign in
  with `az login`. Alternatively, set `AZURE_SPEECH_KEY` to a key from the Speech
  resource. Regional Entra authentication also needs `AZURE_SPEECH_RESOURCE_ID`.

Run the app:

```powershell
python app.py
```

Open <http://127.0.0.1:7860>, select the spoken language, record or upload a WAV
or MP3 clip. Gradio converts uploads to WAV before transcription.
Stop recording, then press **Transcribe audio**. The recognized text appears
in the editable **Text** area, replacing its previous contents.
After successful transcription, the audio recorder and previous generated audio
clear automatically. If the request fails, the recording and existing text are
kept. Press **Clear** to reset the text, recording, and generated audio.

The **Use Speech containers** checkbox currently locks both transcription and
speech synthesis to the configured Speech containers (on). The underlying Speech
functions also support online Azure Speech when called with container mode off.
Configure the container endpoints below before using container mode; missing
endpoints are reported when the corresponding language or voice is used.
The former `USE_CONTAINER` environment variable no longer controls the mode.

The page header shows a compact, rounded green/red status badge for each STT/TTS
container beside the title. Hover over a badge to see the readiness details.
Status is checked when the page opens and automatically every 30 seconds while
it remains open, regardless of the selected Speech mode. The Gradio server calls
`<container-endpoint>/ready` with a 3-second timeout per container. Green means
the HTTP request succeeded and the JSON contains `"ready": "ready"`; any other
status, connection error, invalid response, or missing endpoint is shown in red
with an explanation in the tooltip. These indicators monitor containers, not
online Azure Speech.

Below the text area, select a multilingual voice and click **Convert text to speech**
to convert its current contents to audio, including any edits or manually entered
text. Empty or whitespace-only text is rejected. **Generated audio** starts playback
automatically when audio is ready. If your browser blocks autoplay, press Play.
TTS uses the selected Speech mode, and is never called automatically.
The audio player clears after a successful transcription or when **Clear** is pressed.
Long texts are synthesized sequentially in chunks of at most 1,000 characters,
preferably split at line or word boundaries, to avoid long-request failures in
Speech containers. The PCM audio is joined in order into a single recording.
If any chunk fails, the request reports an error instead of returning partial audio.

## Authentication and permissions

Speech uses `AZURE_SPEECH_KEY` when configured; otherwise it uses the existing
Azure CLI / `DefaultAzureCredential` chain. With a custom Speech endpoint, the
SDK receives the credential directly. For regional Entra authentication, set the
full `AZURE_SPEECH_RESOURCE_ID` copied from the Speech resource's Properties page;
the SDK then receives `aad#<resource-id>#<access-token>`.
The Speech resource must have a custom subdomain configured for Entra authentication.
Do not pass a regional `https://<region>.api.cognitive.microsoft.com/` URL with
`token_credential`: Azure rejects this combination with HTTP 400. Use your
resource-specific custom endpoint, or a Speech resource key for that regional URL.

For local development, sign in with `az login`. In Azure Container Apps, enable
a system-assigned or user-assigned managed identity and grant it
**Cognitive Services Speech User** on the Azure AI Speech resource.

Set `AZURE_CLIENT_ID` only when the app should use a particular user-assigned
managed identity. Keep resource keys and credentials in environment variables or
a managed secret store; never commit them to source control.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `AZURE_SPEECH_ENDPOINT` | Dedicated Speech endpoint; takes precedence over region | *(unset)* |
| `AZURE_SPEECH_REGION` | Speech region when endpoint is unset | *(required without endpoint)* |
| `AZURE_SPEECH_KEY` | Optional Speech resource key; omit to use Entra ID | *(unset)* |
| `AZURE_SPEECH_RESOURCE_ID` | Full Speech ARM resource ID for regional Entra authentication | *(required without key or custom endpoint)* |
| `CONTAINER_STT_EN_ENDPOINT` | Speech container host for English recognition | *(unset)* |
| `CONTAINER_STT_SL_ENDPOINT` | Speech container host for Slovenian recognition | *(unset)* |
| `CONTAINER_TTS_SERAPHINA_ENDPOINT` | Speech container host for the Seraphina voice | *(unset)* |
| `CONTAINER_TTS_ANDREW_ENDPOINT` | Speech container host for the Andrew voice | *(unset)* |
| `AZURE_CLIENT_ID` | Optional user-assigned managed identity client ID | *(unset)* |
| `GRADIO_HOST` | Bind address | `127.0.0.1` |
| `GRADIO_PORT` | Port | `7860` |

The UI supports English (US) and Slovenian for recognition.

## Azure deployment

The included `azure.yaml` targets Azure Container Apps. Configure the environment
variables above for the deployed app and grant its managed identity the required
resource roles. For hosted microphone access, use the HTTPS URL provided by
Container Apps.

After changing the Dockerfile, rebuild and redeploy the app image so the running
container includes FFmpeg and `ffprobe`.
