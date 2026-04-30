# gpt-realtime Speech-to-Speech Demo

A minimal, single-file speech-to-speech demo using the **`gpt-realtime-1.5`** model on **Azure AI Foundry**.
The UI is built with **Gradio** and real-time bidirectional audio is handled by **FastRTC** (WebRTC).
No separate Flask/API layer — everything runs in one process.

## Architecture

```
Browser mic ──WebRTC──▶ Gradio + FastRTC ──WebSocket──▶ Azure Foundry gpt-realtime
Browser speaker ◀─WebRTC── Gradio + FastRTC ◀──WebSocket── Azure Foundry gpt-realtime
```

- **Gradio** serves the web UI
- **FastRTC** streams 24 kHz PCM16 audio frames over WebRTC
- The handler forwards frames to the Foundry realtime WebSocket via `openai` SDK
- Server-side VAD (voice activity detection) drives turn-taking

## Prerequisites

- Python 3.10+
- An Azure AI Foundry deployment of `gpt-realtime` (e.g. `gpt-realtime-1.5`)
- Azure CLI (`az`) for local sign-in, **or** a Managed Identity when running on Azure
- A TURN provider for non-localhost WebRTC (Cloudflare TURN is used in this sample)
- The runtime identity used by the app must have the **`Cognitive Services OpenAI User`** role on the Foundry resource

## Azure Quick Checklist

Before testing the deployed URL, verify all of these:

1. The app is opened over `https://` (not `http://`).
2. Container App env vars include `CLOUDFLARE_TURN_KEY_ID` and `CLOUDFLARE_TURN_KEY_API_TOKEN`.
3. If `AZURE_CLIENT_ID` is set, that exact user-assigned identity has `Cognitive Services OpenAI User` on the Azure OpenAI resource.
4. If `AZURE_CLIENT_ID` is not set, the system-assigned identity has `Cognitive Services OpenAI User`.
5. After any role assignment changes, wait 1-3 minutes for RBAC propagation.
6. The deployed code uses FastRTC parameters `turn_key_id` and `turn_key_api_token`.

## Authentication

This app uses **`DefaultAzureCredential`** — no API keys. Credentials are resolved in this order:

1. Environment variables (`AZURE_CLIENT_ID` / `AZURE_TENANT_ID` / `AZURE_CLIENT_SECRET`)
2. Managed Identity (when deployed on Azure)
3. Azure CLI (`az login`)
4. Azure PowerShell, VS Code, etc.

For local dev just run `az login` once.

## Setup

```powershell
# 1. Sign in to Azure
az login

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
Copy-Item .env.example .env
# edit .env and set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT
```

## Run

```powershell
python app.py
```

Open <http://127.0.0.1:7860> in a modern browser (Chrome/Edge/Firefox), allow microphone access, click **Record**, and start talking.

Use the **Voice** dropdown in the UI to choose which voice the model should use.

## Configuration

All settings live in `.env`:

| Variable | Description | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Foundry resource endpoint | *(required)* |
| `AZURE_OPENAI_DEPLOYMENT` | Realtime deployment name | `gpt-realtime-1.5` |
| `AZURE_OPENAI_API_VERSION` | API version | `2025-04-01-preview` |
| `AZURE_CLIENT_ID` | Optional: force a specific user-assigned managed identity in Azure | *(unset)* |
| `AZURE_TENANT_ID` | Optional tenant hint for credential resolution | *(unset)* |
| `SYSTEM_INSTRUCTIONS` | System prompt | friendly assistant |
| `CLOUDFLARE_TURN_KEY_ID` | Cloudflare TURN key id for WebRTC relay | *(required in Azure)* |
| `CLOUDFLARE_TURN_KEY_API_TOKEN` | Cloudflare TURN API token for WebRTC relay | *(required in Azure)* |
| `GRADIO_HOST` | Bind address | `127.0.0.1` |
| `GRADIO_PORT` | Port | `7860` |

## Azure Deployment Notes

For Azure Container Apps, these are the key points that make this app work reliably:

1. TURN is required for browser-to-server audio relay.
2. HTTPS must be used in production (secure context is required for mic + WebRTC).
3. The identity actually used by `DefaultAzureCredential` must have OpenAI RBAC.

### TURN setup (Cloudflare)

Set both values in your Container App environment:

- `CLOUDFLARE_TURN_KEY_ID`
- `CLOUDFLARE_TURN_KEY_API_TOKEN`

Without TURN, connections often fail in Azure with WebRTC initialization errors.

### Managed Identity and RBAC

This app authenticates to Azure OpenAI with Microsoft Entra ID.

- If `AZURE_CLIENT_ID` is set, `DefaultAzureCredential` targets that user-assigned managed identity.
- If `AZURE_CLIENT_ID` is not set, the platform may use system-assigned identity.

Grant **`Cognitive Services OpenAI User`** on the Azure OpenAI resource to the identity that is actually selected.

If your Container App has both system-assigned and user-assigned identities, verify which one is active and assign role accordingly.

### Container App identity configuration patterns

Use one of these two patterns and keep it explicit. Most `HTTP 401` websocket failures in this app come from identity mismatch (role assigned to one identity, but runtime uses the other).

#### Pattern A: system-assigned identity only

Use this when you do not need a reusable user-assigned identity.

1. Enable only system-assigned identity on the Container App.
2. Ensure `AZURE_CLIENT_ID` is not set in Container App environment variables.
3. Assign `Cognitive Services OpenAI User` to the system-assigned identity on the Azure OpenAI resource scope.

Example commands:

```powershell
# 1) Confirm identity mode and get system-assigned principalId
az containerapp show -g <rg> -n <app> --query "identity" -o json

# 2) Ensure AZURE_CLIENT_ID is absent for this mode
az containerapp show -g <rg> -n <app> --query "properties.template.containers[0].env[].name" -o tsv

# 3) Assign OpenAI role to system-assigned principal
az role assignment create \
	--assignee <systemAssignedPrincipalId> \
	--role "Cognitive Services OpenAI User" \
	--scope <azureOpenAiResourceId>
```

#### Pattern B: both system-assigned and user-assigned identities enabled

Use this when you want a stable, reusable user-assigned identity across resources.

1. Enable both identity types on the Container App.
2. Set `AZURE_CLIENT_ID` to the user-assigned identity client id. This pins `DefaultAzureCredential` to that identity.
3. Assign `Cognitive Services OpenAI User` to that user-assigned identity principal on the Azure OpenAI resource scope.

Example commands:

```powershell
# 1) List user-assigned identities attached to the app
az containerapp show -g <rg> -n <app> --query "identity.userAssignedIdentities" -o json

# 2) Ensure AZURE_CLIENT_ID matches the intended user-assigned clientId
az containerapp show -g <rg> -n <app> --query "properties.template.containers[0].env[?name=='AZURE_CLIENT_ID']" -o json

# 3) Assign OpenAI role to user-assigned principal
az role assignment create \
	--assignee <userAssignedPrincipalId> \
	--role "Cognitive Services OpenAI User" \
	--scope <azureOpenAiResourceId>
```

Operational notes:

- If `AZURE_CLIENT_ID` is set, the app will use that user-assigned identity and ignore system-assigned identity for token acquisition.
- Wait 1-3 minutes after new role assignments for RBAC propagation.
- If uncertain, temporarily assign role to both identities, validate traffic, then remove the unused assignment.

## Troubleshooting (Known Errors)

### 1) `Connection failed` during RTC startup

Most common cause: missing TURN credentials in Azure.

Check:

- `CLOUDFLARE_TURN_KEY_ID` and `CLOUDFLARE_TURN_KEY_API_TOKEN` are present on the Container App.
- Browser is using HTTPS URL.

### 2) `server rejected WebSocket connection: HTTP 401`

This is Azure OpenAI auth failure from the server-side websocket connect.

Check:

- The Container App managed identity has `Cognitive Services OpenAI User` on the target OpenAI account.
- If `AZURE_CLIENT_ID` is set, make sure that specific user-assigned identity has the role.
- Wait 1-3 minutes after role assignment for RBAC propagation.

### 3) `get_cloudflare_turn_credentials() got an unexpected keyword argument 'key_id'`

This indicates an API mismatch with installed FastRTC version.

Use:

- `turn_key_id`
- `turn_key_api_token`

instead of `key_id` / `api_token`.

## Notes

- The model uses 24 kHz mono PCM16 audio in both directions.
- Turn detection is server-side VAD; adjust `threshold`/`silence_duration_ms` in `app.py` if needed.
- For production, put the app behind HTTPS (WebRTC requires a secure context on non-localhost hosts).
- Keep credentials server-side — the browser never sees a token; it only speaks WebRTC to the Python server.
- For Azure Container Apps, configure TURN credentials and verify the correct managed identity has `Cognitive Services OpenAI User` on the Azure OpenAI resource.
