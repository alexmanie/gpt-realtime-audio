# Azure Speech Assistant

A Gradio conversation using **Azure AI Speech** for speech-to-text (STT) and
**Azure OpenAI** for automatic generative replies, with on-demand
**Azure AI Speech** text-to-speech (TTS).

## Architecture

```text
Browser microphone -> Gradio server -> Azure AI Speech STT endpoint
Conversation dialog <- recognized text (user message) <-
Transcript + conversation history -> Azure OpenAI -> assistant message
```

Each recorded turn is uploaded to the Python app as a WAV file. The server
sends it to the configured Speech endpoint, collects all recognized sentences,
and appends the transcription as a user message in the conversation.
It then automatically sends the conversation history and `SYSTEM_INSTRUCTIONS.md`
to the configured Azure OpenAI chat deployment and appends its answer.
The transcript is shown before the LLM call and remains visible if that call fails.
Messages are isolated to the current Gradio session. Recognition errors are
shown in the frontend without adding an empty or partial message.
WebRTC and TURN credentials are not required.

Browser recordings are converted to 16 kHz, mono, signed 16-bit PCM before
being sent to Speech. Empty or silent recordings are rejected locally.
If recognition returns no text, the frontend shows Azure's no-match reason,
the selected language, and audio duration/level details. Play the recording
back to verify the microphone captured speech and check the language selector.

## Prerequisites

- Python 3.10+
- An Azure AI Speech resource and its endpoint (or region)
- An Azure OpenAI resource with a chat-capable deployment
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

- `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_CHAT_DEPLOYMENT` to the OpenAI
  resource endpoint and an existing chat deployment name. The name is displayed
  in the UI; there is no implicit default model.
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
clip. Stop recording, then press **Send**. The recognized text appears
as a user message in **Conversation**, followed automatically by the LLM's reply.
Once the reply arrives, the audio recorder clears automatically for the next
message without clearing the conversation. If the request fails, the recording
is kept.
Repeat to add another message, or press
**Clear conversation** to reset the dialog.

Use the **Use Speech containers** checkbox to switch between online Azure Speech
(off, the default) and the configured Speech containers (on). The selection is
per browser session and applies to both transcription and speech synthesis.
Switching modes does not change the conversation or the Azure OpenAI service.
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
with an explanation in the tooltip. These indicators monitor containers, not online Azure Speech
or Azure OpenAI.

Below the conversation, select a multilingual voice and click **Speak last reply**
to convert the latest assistant message to audio. **Spoken reply** starts playback
automatically when audio is ready. If your browser blocks autoplay, press Play.
TTS uses the selected Speech mode, and is never called automatically.
The audio player clears when a new OpenAI reply
arrives or the conversation is cleared.

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
Also grant **Cognitive Services OpenAI User** on the Azure OpenAI resource.

Set `AZURE_CLIENT_ID` only when the app should use a particular user-assigned
managed identity. Keep resource keys and credentials in environment variables or
a managed secret store; never commit them to source control.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint | *(required)* |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Existing chat deployment name | *(required)* |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version | `2025-04-01-preview` |
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
