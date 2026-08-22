"""FastAPI Local Web Server & Multi-Session Dashboard with Dropdown Presets & Encrypted Local Vault."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT
from engine import ConversationEngine
from commands import handle_command
from validation import CaliperValidationGate
from exceptions import ChatbotError, EmptyInputError, InputLengthError, InvalidCommandError
from security import (
    get_key_status,
    save_vault_api_settings,
    get_vault_api_settings_summary,
    resolve_api_config,
    resolve_api_key,
    decrypt_credential,
)
from persistence import SessionStore
from llm_client import LLMClient

# Initialize FastAPI App
app = FastAPI(
    title="Stateful Conversational Agent",
    description="DecodeLabs Project 1 — Dropdown Presets, In-App Encrypted Vault & Dual Memory Interface",
    version="2.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistence store
_store = SessionStore()

# In-memory engine cache keyed by session_id: Dict[session_id, ConversationEngine]
_active_engines: Dict[str, ConversationEngine] = {}


def get_or_create_engine(session_id: Optional[str] = None) -> ConversationEngine:
    """
    Retrieves an existing ConversationEngine instance or restores it from SQLite persistence.
    If session_id is None, creates a fresh session.
    """
    global _active_engines

    if session_id and session_id in _active_engines:
        engine = _active_engines[session_id]
        engine.reload_global_facts()
        return engine

    engine = ConversationEngine(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    if session_id:
        # Attempt to resume from persistence
        loaded = engine.load_session(session_id)
        if not loaded:
            # If not in persistence, start new session with this id
            engine.session_id = session_id
        engine.reload_global_facts()
        _active_engines[session_id] = engine
        return engine
    else:
        # Start a brand new session
        new_sid = engine.start_session()
        engine.reload_global_facts()
        _active_engines[new_sid] = engine
        return engine


# ──────────────────────────── Request / Response Schemas ──────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class CommandRequest(BaseModel):
    command: str
    args: str = ""
    session_id: Optional[str] = None


class FactRequest(BaseModel):
    key: str
    value: str


class SettingsRequest(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key: Optional[str] = None


class TestConnectionRequest(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key: Optional[str] = None


# ──────────────────────────── API Endpoints ──────────────────────────

@app.get("/api/health")
async def health_check():
    config = resolve_api_config()
    return {"status": "ok", "model": config["model"], "provider": config["provider"]}


# ──────────────────────────── In-App Settings & Vault Endpoints ──────────────────────────

@app.get("/api/settings")
async def get_settings_endpoint():
    """Returns safe summary of current API settings (masked key, provider, base_url, model)."""
    return get_vault_api_settings_summary()


@app.post("/api/settings")
async def save_settings_endpoint(payload: SettingsRequest):
    """Encrypts and saves API settings into the local SQLite vault and updates active engines."""
    provider = payload.provider.strip().lower()
    base_url = payload.base_url.strip()
    model = payload.model.strip()
    api_key = payload.api_key.strip() if payload.api_key else None

    if not base_url or not model:
        raise HTTPException(status_code=400, detail="Base URL and Model Name are required.")

    # Save to encrypted vault
    save_vault_api_settings(provider=provider, base_url=base_url, model=model, api_key=api_key)

    # Reconfigure all cached engines
    for engine in _active_engines.values():
        engine.configure(provider=provider, base_url=base_url, model=model, api_key=api_key)

    return {
        "status": "saved",
        "message": "Settings encrypted and saved to local database vault!",
        "settings": get_vault_api_settings_summary(),
    }


@app.post("/api/settings/test")
async def test_connection_endpoint(payload: TestConnectionRequest):
    """Tests connection against the specified OpenAI-compatible endpoint with a lightweight probe."""
    base_url = payload.base_url.strip()
    model = payload.model.strip()
    api_key = payload.api_key.strip() if payload.api_key else ""

    # If key wasn't supplied in test payload, try to resolve existing saved key
    if not api_key:
        try:
            api_key = resolve_api_key(interactive_fallback=False)
        except Exception:
            raise HTTPException(status_code=400, detail="Please enter an API key to test connection.")

    result = LLMClient.test_connection(base_url=base_url, api_key=api_key, model=model)
    return result


# ──────────────────────────── Sessions Endpoints ──────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """Returns list of all saved sessions with metadata and titles."""
    return _store.list_sessions(limit=50)


@app.post("/api/sessions")
async def create_session():
    """Creates a new isolated chat session."""
    engine = ConversationEngine(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    session_id = engine.start_session()
    _active_engines[session_id] = engine
    return {
        "session_id": session_id,
        "model": engine.model,
        "created_at": engine._created_at.isoformat() if engine._created_at else None,
    }


@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Retrieves session metadata, conversation history, and telemetry."""
    session = _store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    engine = get_or_create_engine(session_id)
    messages = _store.get_session_messages(session_id)
    stats = engine.get_stats()
    local_facts = engine.get_local_facts()

    return {
        "session": session,
        "messages": messages,
        "stats": stats,
        "local_facts": local_facts,
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Permanently deletes a chat session and all its messages from SQLite."""
    deleted = _store.delete_session(session_id)
    if session_id in _active_engines:
        del _active_engines[session_id]
    return {"deleted": deleted, "session_id": session_id}


# ──────────────────────────── Local Memory Endpoints ──────────────────────────

@app.get("/api/sessions/{session_id}/local-memory")
async def get_local_memory_endpoint(session_id: str):
    """Retrieves all local pinned facts for a specific session."""
    engine = get_or_create_engine(session_id)
    return engine.get_local_facts()


@app.post("/api/sessions/{session_id}/local-memory")
async def set_local_memory_endpoint(session_id: str, payload: FactRequest):
    """Pins a fact specifically for this session."""
    key = payload.key.strip()
    val = payload.value.strip()
    if not key or not val:
        raise HTTPException(status_code=400, detail="Key and Value are required.")

    engine = get_or_create_engine(session_id)
    engine.remember_fact(key, val)
    return {
        "status": "pinned",
        "key": key,
        "value": val,
        "facts": engine.get_local_facts(),
        "stats": engine.get_stats(),
    }


@app.delete("/api/sessions/{session_id}/local-memory/{key}")
async def delete_local_memory_endpoint(session_id: str, key: str):
    """Removes a local pinned fact from this session."""
    engine = get_or_create_engine(session_id)
    removed = engine.forget_fact(key)
    return {
        "removed": removed,
        "key": key,
        "facts": engine.get_local_facts(),
        "stats": engine.get_stats(),
    }


# ──────────────────────────── Global Memory Endpoints ──────────────────────────

@app.get("/api/global-memory")
async def get_global_memory():
    """Retrieves all global memory facts shared across conversations."""
    return _store.get_all_global_facts_detailed()


@app.post("/api/global-memory")
async def set_global_memory(payload: FactRequest):
    """Adds or updates a global memory fact available to all sessions."""
    key = payload.key.strip()
    val = payload.value.strip()
    if not key or not val:
        raise HTTPException(status_code=400, detail="Key and Value are required.")

    _store.set_global_fact(key, val)
    # Sync all active engines
    for engine in _active_engines.values():
        engine.reload_global_facts()

    return {"status": "saved", "key": key, "value": val}


@app.delete("/api/global-memory/{key}")
async def delete_global_memory(key: str):
    """Deletes a global memory fact."""
    deleted = _store.delete_global_fact(key)
    # Sync all active engines
    for engine in _active_engines.values():
        engine.reload_global_facts()

    return {"deleted": deleted, "key": key}


@app.get("/api/key-status")
async def key_status_endpoint():
    return get_key_status()


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    raw_text = payload.message.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    engine = get_or_create_engine(payload.session_id)

    # Check if input is a slash command
    is_command, cmd_name, cmd_args = CaliperValidationGate.parse_and_validate_command(raw_text)
    if is_command:
        try:
            result_text, should_exit = handle_command(cmd_name, cmd_args, engine)
            return {
                "type": "command",
                "content": result_text,
                "session_id": engine.session_id,
                "stats": engine.get_stats(),
                "local_facts": engine.get_local_facts(),
            }
        except InvalidCommandError as e:
            raise HTTPException(status_code=400, detail=e.message)
        except ChatbotError as e:
            raise HTTPException(status_code=500, detail=e.message)

    # Conversational turn
    try:
        response = engine.send_message(raw_text)
        return {
            "type": "chat",
            "content": response.content,
            "latency_sec": response.latency_sec,
            "usage": response.usage.model_dump(),
            "model": response.model,
            "session_id": engine.session_id,
            "stats": engine.get_stats(),
            "local_facts": engine.get_local_facts(),
        }
    except EmptyInputError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except InputLengthError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except ChatbotError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


# ──────────────────────────── Single-Page Web UI ──────────────────────────

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stateful Conversational Agent — DecodeLabs</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    body { font-family: 'Inter', sans-serif; }
    pre, code { font-family: 'JetBrains Mono', monospace; }
    .prose pre { background-color: #18181b; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; color: #e4e4e7; margin: 0.75rem 0; font-size: 0.875rem; border: 1px solid #27272a; }
    .prose p { margin-bottom: 0.5rem; }
    .prose ul, .prose ol { margin-left: 1.25rem; margin-bottom: 0.5rem; }
    .prose code:not(pre code) { background-color: #27272a; padding: 0.125rem 0.375rem; border-radius: 0.25rem; color: #38bdf8; font-size: 0.875rem; }
    .custom-scrollbar::-webkit-scrollbar { width: 5px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #27272a; border-radius: 3px; }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
    select { background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%2371717a' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e"); background-position: right 0.5rem center; background-repeat: no-repeat; background-size: 1.5em 1.5em; padding-right: 2rem; -webkit-appearance: none; -moz-appearance: none; appearance: none; }
  </style>
</head>
<body class="bg-zinc-950 text-zinc-100 flex h-screen overflow-hidden antialiased">

  <!-- Sidebar -->
  <aside class="w-80 bg-zinc-900/95 border-r border-zinc-800 flex flex-col justify-between hidden md:flex shrink-0">
    <div class="flex-1 flex flex-col min-h-0">
      <!-- App Header -->
      <div class="p-4 border-b border-zinc-800/80 flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <i class="fa-solid fa-brain text-white text-lg"></i>
          </div>
          <div>
            <h1 class="font-bold text-sm text-zinc-100 leading-tight">Stateful Agent</h1>
            <p class="text-[11px] text-zinc-400">DecodeLabs · Multi-Session</p>
          </div>
        </div>
      </div>

      <!-- Navigation Tabs: Chats vs Global Memory vs Settings -->
      <div class="px-3 pt-3">
        <div class="grid grid-cols-3 gap-1 bg-zinc-950 p-1 rounded-xl border border-zinc-800 text-xs font-medium">
          <button
            id="tab-btn-chats"
            onclick="switchSidebarTab('chats')"
            class="py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 bg-zinc-800 text-white shadow-sm"
          >
            <i class="fa-regular fa-message text-xs text-cyan-400"></i>
            <span>Chats</span>
          </button>
          <button
            id="tab-btn-global"
            onclick="switchSidebarTab('global')"
            class="py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 text-zinc-400 hover:text-zinc-200"
          >
            <i class="fa-solid fa-earth-americas text-xs text-amber-400"></i>
            <span>Global</span>
          </button>
          <button
            id="tab-btn-settings"
            onclick="switchSidebarTab('settings')"
            class="py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 text-zinc-400 hover:text-zinc-200"
          >
            <i class="fa-solid fa-gear text-xs text-purple-400"></i>
            <span>Settings</span>
          </button>
        </div>
      </div>

      <!-- TAB 1: Chats List View -->
      <div id="tab-content-chats" class="flex-1 flex flex-col min-h-0">
        <!-- New Chat Button -->
        <div class="p-3 border-b border-zinc-800/40">
          <button
            onclick="createNewSession()"
            class="w-full py-2.5 px-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl font-medium text-xs shadow-md shadow-cyan-600/20 transition-all flex items-center justify-center gap-2 group"
          >
            <i class="fa-solid fa-plus text-xs group-hover:rotate-90 transition-transform"></i>
            <span>New Chat Session</span>
          </button>
        </div>

        <!-- Session List (Scrollable) -->
        <div class="flex-1 overflow-y-auto p-3 space-y-1.5 custom-scrollbar min-h-0">
          <div class="flex items-center justify-between px-2 py-1 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
            <span>Conversations</span>
            <span id="total-sessions-badge" class="text-[10px] bg-zinc-800 text-zinc-400 px-1.5 py-0.2 rounded font-mono">0</span>
          </div>
          <div id="session-list" class="space-y-1">
            <div class="text-xs text-zinc-500 italic p-3 text-center">Loading conversations...</div>
          </div>
        </div>
      </div>

      <!-- TAB 2: Global Memory View -->
      <div id="tab-content-global" class="flex-1 flex flex-col min-h-0 p-3 space-y-3 hidden">
        <!-- Global Memory Header & Explanation -->
        <div class="bg-amber-950/20 border border-amber-800/30 rounded-xl p-3 text-xs space-y-1.5">
          <div class="flex items-center gap-1.5 font-semibold text-amber-400">
            <i class="fa-solid fa-earth-americas"></i>
            <span>Global Memory Layer</span>
          </div>
          <p class="text-zinc-300 text-[11px] leading-relaxed">
            Facts added here are <b>shared across all chat sessions</b> and automatically injected into every conversation's system context.
          </p>
        </div>

        <!-- Add Global Fact Form -->
        <form onsubmit="handleAddGlobalFact(event)" class="bg-zinc-950/70 border border-zinc-800 rounded-xl p-3 space-y-2.5">
          <p class="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">Add Global Fact</p>

          <!-- Preset Dropdown -->
          <div class="space-y-1">
            <label class="text-[10px] text-zinc-500 block">Template Preset:</label>
            <select
              onchange="handleGlobalPresetSelect(this)"
              class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:border-amber-500 focus:outline-none cursor-pointer"
            >
              <option value="">-- Choose a Preset Template --</option>
              <option value="Tone|Concise, clear, and direct">Tone: Concise & direct</option>
              <option value="Role|Generative AI Engineer">Role: Generative AI Engineer</option>
              <option value="Preferred Language|Python 3.12+ / TypeScript">Language: Python 3.12+ / TypeScript</option>
              <option value="Code Style|Strict typing, docstrings, clean architecture">Code Style: Strict typing & docstrings</option>
              <option value="Response Format|Markdown with code blocks & step-by-step reasoning">Format: Markdown with reasoning</option>
            </select>
          </div>

          <input
            id="global-key-input"
            type="text"
            placeholder="Key (e.g. Tone, Language, Role)..."
            class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-amber-500 focus:outline-none"
            required
          />
          <textarea
            id="global-val-input"
            rows="2"
            placeholder="Value (e.g. Concise & technical, Python 3.12+, AI Engineer)..."
            class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-amber-500 focus:outline-none resize-none"
            required
          ></textarea>

          <button
            type="submit"
            class="w-full py-1.5 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-medium shadow-sm transition-colors flex items-center justify-center gap-1.5"
          >
            <i class="fa-solid fa-thumbtack text-xs"></i>
            <span>Save Global Fact</span>
          </button>
        </form>

        <!-- Global Facts List (Scrollable) -->
        <div class="flex-1 overflow-y-auto space-y-1.5 custom-scrollbar min-h-0">
          <p class="text-[11px] font-medium text-zinc-500 uppercase tracking-wider px-1">Active Global Facts</p>
          <div id="global-facts-list" class="space-y-1.5">
            <!-- Populated by JS -->
          </div>
        </div>
      </div>

      <!-- TAB 3: Settings & Frontier Presets View (Dropdown-Based) -->
      <div id="tab-content-settings" class="flex-1 flex flex-col min-h-0 p-3 space-y-3 hidden overflow-y-auto custom-scrollbar">
        <!-- Settings Header -->
        <div class="bg-purple-950/20 border border-purple-800/30 rounded-xl p-3 text-xs space-y-1.5">
          <div class="flex items-center justify-between font-semibold text-purple-400">
            <span class="flex items-center gap-1.5"><i class="fa-solid fa-shield-halved"></i> Frontier AI Settings</span>
            <span class="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.2 rounded font-mono">AES Vault</span>
          </div>
          <p class="text-zinc-300 text-[11px] leading-relaxed">
            Configure any frontier model. Keys are encrypted locally using <b>AES Fernet</b> in SQLite.
          </p>
        </div>

        <!-- Settings Form with Dropdowns -->
        <form onsubmit="handleSaveSettings(event)" class="bg-zinc-950/70 border border-zinc-800 rounded-xl p-3 space-y-3">
          <!-- 1. Provider Preset Dropdown -->
          <div class="space-y-1">
            <label class="text-[11px] font-semibold text-zinc-300 uppercase tracking-wider block">AI Provider Preset</label>
            <select
              id="settings-provider-select"
              onchange="handleProviderSelectChange(this.value)"
              class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-2 text-xs text-zinc-100 focus:border-purple-500 focus:outline-none cursor-pointer font-medium"
            >
              <option value="nvidia">🟢 NVIDIA NIM (Default / Free Endpoints)</option>
              <option value="openai">🟢 OpenAI (ChatGPT / GPT-4o / o1 / o3-mini)</option>
              <option value="claude">🟣 Anthropic (Claude 3.7 / 3.5 Sonnet)</option>
              <option value="deepseek">🔵 DeepSeek Official (DeepSeek-V3 / R1)</option>
              <option value="qwen">🟣 Alibaba Qwen (DashScope / Qwen-Max)</option>
              <option value="gemini">🔵 Google Gemini (Gemini 2.5 Flash / Pro)</option>
              <option value="groq">🟠 Groq LPU (Ultra-Fast Llama 3.3)</option>
              <option value="openrouter">🌐 OpenRouter (Universal 200+ Models)</option>
              <option value="custom">⚪ Custom OpenAI-Compatible (Ollama / Local)</option>
            </select>
          </div>

          <input type="hidden" id="settings-provider-input" value="nvidia" />

          <!-- 2. Base URL Input -->
          <div class="space-y-1">
            <label class="text-[11px] font-medium text-zinc-400">API Base URL</label>
            <input
              id="settings-baseurl-input"
              type="text"
              placeholder="https://integrate.api.nvidia.com/v1"
              class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:border-purple-500 focus:outline-none font-mono"
              required
            />
          </div>

          <!-- 3. Model Name Suggestion Dropdown & Input -->
          <div class="space-y-1.5">
            <label class="text-[11px] font-medium text-zinc-400">Top Model Suggestions</label>
            <select
              id="settings-model-select"
              onchange="handleModelSelectChange(this.value)"
              class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-purple-300 focus:border-purple-500 focus:outline-none cursor-pointer font-mono"
            >
              <!-- Populated by JS -->
            </select>

            <div class="space-y-1 pt-1">
              <label class="text-[10px] text-zinc-500 block">Exact Model ID / Name:</label>
              <input
                id="settings-model-input"
                type="text"
                placeholder="meta/llama-3.1-8b-instruct"
                class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-600 focus:border-purple-500 focus:outline-none font-mono"
                required
              />
            </div>
          </div>

          <!-- 4. API Key Input with Eye Toggle -->
          <div class="space-y-1">
            <div class="flex justify-between items-center">
              <label class="text-[11px] font-medium text-zinc-400">API Key</label>
              <span id="key-configured-badge" class="text-[10px] text-zinc-500 font-mono">Loading...</span>
            </div>
            <div class="relative">
              <input
                id="settings-key-input"
                type="password"
                placeholder="Enter new key to update or replace..."
                class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 pr-8 text-xs text-zinc-100 placeholder-zinc-600 focus:border-purple-500 focus:outline-none font-mono"
              />
              <button
                type="button"
                onclick="toggleKeyVisibility()"
                class="absolute right-2 top-2 text-zinc-500 hover:text-zinc-300 text-xs"
              >
                <i id="key-visibility-icon" class="fa-solid fa-eye"></i>
              </button>
            </div>
            <p class="text-[10px] text-zinc-500">Leave blank to keep existing encrypted key.</p>
          </div>

          <!-- Test & Save Buttons -->
          <div class="space-y-2 pt-1">
            <button
              type="button"
              id="btn-test-connection"
              onclick="handleTestConnection()"
              class="w-full py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-medium transition-all flex items-center justify-center gap-1.5"
            >
              <i class="fa-solid fa-plug text-xs text-cyan-400"></i>
              <span>Test Connection</span>
            </button>
            
            <button
              type="submit"
              class="w-full py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-lg text-xs font-semibold shadow-md shadow-purple-600/20 transition-all flex items-center justify-center gap-1.5"
            >
              <i class="fa-solid fa-lock text-xs"></i>
              <span>Encrypt & Save Settings</span>
            </button>
          </div>

          <!-- Test Result Box -->
          <div id="test-result-box" class="hidden p-2.5 rounded-lg text-xs border"></div>
        </form>
      </div>

      <!-- Telemetry & Headroom Card (Shared) -->
      <div class="p-3 border-t border-zinc-800/80 bg-zinc-950/40 space-y-2.5 shrink-0">
        <div class="bg-zinc-900/90 border border-zinc-800/80 rounded-xl p-3 space-y-2">
          <div class="flex justify-between items-center text-xs">
            <span class="text-zinc-400 flex items-center gap-1.5 text-[11px] font-medium"><i class="fa-solid fa-gauge text-cyan-400"></i> Context Headroom</span>
            <span id="token-ratio" class="font-mono text-zinc-300 text-[11px]">0 / 8,192</span>
          </div>
          <!-- Progress bar -->
          <div class="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
            <div id="token-bar" class="bg-gradient-to-r from-cyan-500 to-blue-500 h-1.5 rounded-full transition-all duration-300" style="width: 2%"></div>
          </div>
          <div class="grid grid-cols-2 gap-2 text-xs pt-1 border-t border-zinc-800/60 font-mono text-[11px]">
            <div>
              <span class="text-zinc-500 block text-[9px] uppercase tracking-wider">Session Turns</span>
              <span id="stat-turns" class="text-zinc-200 font-semibold">0</span>
            </div>
            <div>
              <span class="text-zinc-500 block text-[9px] uppercase tracking-wider">Global Memory</span>
              <span id="stat-global-count" class="text-amber-400 font-semibold">0</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Active Model & API Status Footer -->
    <div class="p-3 border-t border-zinc-800/80 bg-zinc-950/90 text-xs shrink-0">
      <div class="flex items-center justify-between text-zinc-400 text-[11px]">
        <span class="flex items-center gap-1.5">
          <span id="active-status-dot" class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span id="active-model-display" class="font-mono text-zinc-300 truncate max-w-[150px]">meta/llama-3.1-8b</span>
        </span>
        <span id="active-provider-display" class="text-zinc-500 text-[10px] uppercase font-semibold">NVIDIA</span>
      </div>
      <div class="mt-1 text-[10px] text-zinc-500 font-mono truncate" id="session-id-display">ID: ...</div>
    </div>
  </aside>

  <!-- Main Chat Area -->
  <main class="flex-1 flex flex-col bg-zinc-950 overflow-hidden min-w-0 relative">
    <!-- Top Navigation Bar -->
    <header class="h-14 border-b border-zinc-800/80 px-6 flex items-center justify-between bg-zinc-900/50 backdrop-blur-md shrink-0">
      <div class="flex items-center space-x-3 min-w-0">
        <div class="truncate">
          <h2 id="active-chat-title" class="text-sm font-semibold text-zinc-200 truncate">Current Conversation</h2>
          <p id="active-chat-meta" class="text-[11px] text-zinc-500 font-mono truncate">Ready to chat</p>
        </div>
      </div>
      <div class="flex items-center space-x-2 text-xs">
        <!-- Local Memory Tab Button -->
        <button
          id="btn-local-memory"
          onclick="toggleLocalMemoryDrawer()"
          class="px-3 py-1.5 bg-emerald-950/40 border border-emerald-500/30 hover:bg-emerald-900/50 text-emerald-300 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 shadow-sm"
          title="Manage Pinned Memory for this Conversation"
        >
          <i class="fa-solid fa-thumbtack text-emerald-400"></i>
          <span>Local Memory</span>
          <span id="local-count-badge" class="bg-emerald-500/20 text-emerald-300 px-1.5 py-0.2 rounded-full text-[10px] font-mono">0</span>
        </button>

        <!-- Stats Button -->
        <button
          onclick="runQuickCommand('/stats')"
          class="px-2.5 py-1.5 bg-zinc-800/60 hover:bg-zinc-800 text-zinc-300 rounded-lg text-xs transition-colors flex items-center gap-1.5"
          title="View Statistics"
        >
          <i class="fa-solid fa-chart-pie text-cyan-400"></i>
          <span class="hidden sm:inline">Stats</span>
        </button>

        <!-- Clear RAM Button -->
        <button
          onclick="runQuickCommand('/clear')"
          class="px-2.5 py-1.5 bg-zinc-800/60 hover:bg-zinc-800 text-zinc-300 rounded-lg text-xs transition-colors flex items-center gap-1.5"
          title="Clear In-Memory Rolling Window"
        >
          <i class="fa-solid fa-broom text-amber-400"></i>
          <span class="hidden sm:inline">Clear RAM</span>
        </button>

        <!-- Delete Session Button -->
        <button
          onclick="deleteActiveSession()"
          class="px-2.5 py-1.5 bg-zinc-800/60 hover:bg-red-950/40 hover:text-red-400 text-zinc-300 rounded-lg text-xs transition-colors flex items-center gap-1.5"
          title="Delete This Chat"
        >
          <i class="fa-solid fa-trash text-red-400"></i>
        </button>
      </div>
    </header>

    <!-- Slide-over Drawer for Local Memory -->
    <div
      id="local-memory-drawer"
      class="absolute top-14 right-0 bottom-0 w-84 bg-zinc-900/95 border-l border-zinc-800 p-4 shadow-2xl backdrop-blur-md flex flex-col justify-between transform translate-x-full transition-transform duration-300 z-30 overflow-hidden"
    >
      <div class="flex-1 flex flex-col min-h-0 space-y-3">
        <!-- Drawer Header -->
        <div class="flex items-center justify-between pb-2 border-b border-zinc-800">
          <div class="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <i class="fa-solid fa-thumbtack"></i>
            <span>Local Session Memory</span>
          </div>
          <button onclick="toggleLocalMemoryDrawer()" class="text-zinc-500 hover:text-zinc-300 p-1">
            <i class="fa-solid fa-xmark text-sm"></i>
          </button>
        </div>

        <p class="text-[11px] text-zinc-400 leading-relaxed">
          Facts pinned here apply <b>only to this active chat session</b> and survive context window pruning.
        </p>

        <!-- Add Local Fact Form -->
        <form onsubmit="handleAddLocalFact(event)" class="bg-zinc-950/80 border border-zinc-800 rounded-xl p-3 space-y-2.5 shrink-0">
          <p class="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">Pin Session Fact</p>
          <input
            id="local-key-input"
            type="text"
            placeholder="Key (e.g. Topic, Goal, Codebase)..."
            class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-emerald-500 focus:outline-none"
            required
          />
          <textarea
            id="local-val-input"
            rows="2"
            placeholder="Value (e.g. Building Project 1, React + FastAPI, Target: Production)..."
            class="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 focus:border-emerald-500 focus:outline-none resize-none"
            required
          ></textarea>
          <button
            type="submit"
            class="w-full py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-medium shadow-sm transition-colors flex items-center justify-center gap-1.5"
          >
            <i class="fa-solid fa-thumbtack text-xs"></i>
            <span>Pin to This Chat</span>
          </button>
        </form>

        <!-- Local Facts List (Scrollable) -->
        <div class="flex-1 overflow-y-auto space-y-1.5 custom-scrollbar min-h-0">
          <p class="text-[11px] font-medium text-zinc-500 uppercase tracking-wider px-1">Pinned in this Session</p>
          <div id="local-facts-list" class="space-y-1.5">
            <!-- Populated by JS -->
          </div>
        </div>
      </div>
    </div>

    <!-- Message Scroll Area -->
    <div id="chat-messages" class="flex-1 overflow-y-auto p-6 space-y-5 custom-scrollbar min-h-0">
      <!-- Welcome Message if empty -->
      <div id="welcome-message" class="flex items-start space-x-3 max-w-3xl">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shrink-0 mt-0.5 shadow-md shadow-cyan-600/20">
          <i class="fa-solid fa-robot text-sm"></i>
        </div>
        <div class="bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm px-4 py-3.5 text-sm text-zinc-200 space-y-2 shadow-sm">
          <p class="font-semibold text-cyan-400">Stateful Conversational Agent Ready</p>
          <p class="text-zinc-300 leading-relaxed">
            All conversations are isolated in dedicated sessions and persisted automatically in SQLite.
            Open the <b class="text-purple-400">Settings</b> tab in the sidebar to configure any frontier model (ChatGPT, Claude, DeepSeek, Qwen, Gemini, Groq, NVIDIA).
          </p>
        </div>
      </div>
    </div>

    <!-- Input Bar -->
    <footer class="p-4 border-t border-zinc-800/80 bg-zinc-900/30 shrink-0">
      <form id="chat-form" onsubmit="handleSend(event)" class="max-w-4xl mx-auto flex items-end space-x-3">
        <div class="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-2.5 focus-within:border-cyan-500 focus-within:ring-1 focus-within:ring-cyan-500/50 transition-all flex items-center space-x-2">
          <textarea
            id="message-input"
            rows="1"
            placeholder="Type your message here..."
            class="w-full bg-transparent text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none resize-none leading-relaxed"
            onkeydown="handleKeyDown(event)"
          ></textarea>
        </div>
        <button
          type="submit"
          id="send-btn"
          class="h-11 px-4 bg-cyan-600 hover:bg-cyan-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white rounded-xl font-medium text-sm transition-colors flex items-center justify-center shadow-lg shadow-cyan-600/20 shrink-0"
        >
          <i class="fa-solid fa-paper-plane text-xs mr-1.5"></i> Send
        </button>
      </form>
      <div class="max-w-4xl mx-auto flex justify-between items-center mt-2 px-1 text-[11px] text-zinc-500">
        <span>Press <kbd class="px-1.5 py-0.5 bg-zinc-800 rounded text-zinc-400 font-mono text-[10px]">Enter</kbd> to send, <kbd class="px-1.5 py-0.5 bg-zinc-800 rounded text-zinc-400 font-mono text-[10px]">Shift+Enter</kbd> for newline</span>
        <span id="live-latency" class="font-mono">Ready</span>
      </div>
    </footer>
  </main>

  <script>
    // State
    let activeSessionId = localStorage.getItem('stateful_active_session_id') || null;
    let sessionsCache = [];
    let globalFactsCache = [];
    let localFactsCache = {};
    let isLocalDrawerOpen = false;
    let currentSettings = {};

    const PRESETS = {
      nvidia: {
        provider: 'nvidia',
        name: 'NVIDIA NIM (Default / Free Endpoints)',
        base_url: 'https://integrate.api.nvidia.com/v1',
        model: 'meta/llama-3.1-8b-instruct',
        models: ['meta/llama-3.1-8b-instruct', 'deepseek-ai/deepseek-v4-flash-0731', 'meta/llama-3.3-70b-instruct', 'mistralai/mistral-large-2-instruct']
      },
      openai: {
        provider: 'openai',
        name: 'OpenAI (ChatGPT / GPT-4o / o1)',
        base_url: 'https://api.openai.com/v1',
        model: 'gpt-4o',
        models: ['gpt-4o', 'gpt-4o-mini', 'o1', 'o3-mini', 'gpt-4-turbo']
      },
      claude: {
        provider: 'claude',
        name: 'Anthropic (Claude 3.7 / 3.5 Sonnet)',
        base_url: 'https://openrouter.ai/api/v1',
        model: 'anthropic/claude-3.7-sonnet',
        models: ['anthropic/claude-3.7-sonnet', 'anthropic/claude-3.5-sonnet', 'anthropic/claude-3.5-haiku', 'anthropic/claude-3-opus']
      },
      deepseek: {
        provider: 'deepseek',
        name: 'DeepSeek Official (DeepSeek-V3 / R1)',
        base_url: 'https://api.deepseek.com/v1',
        model: 'deepseek-chat',
        models: ['deepseek-chat', 'deepseek-reasoner']
      },
      qwen: {
        provider: 'qwen',
        name: 'Alibaba Qwen (DashScope / Qwen-Max)',
        base_url: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
        model: 'qwen-max',
        models: ['qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen-2.5-72b-instruct', 'qwq-32b']
      },
      gemini: {
        provider: 'gemini',
        name: 'Google Gemini (Gemini 2.5 Flash / Pro)',
        base_url: 'https://generativelanguage.googleapis.com/v1beta/openai/',
        model: 'gemini-2.5-flash',
        models: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-1.5-pro']
      },
      groq: {
        provider: 'groq',
        name: 'Groq LPU (Ultra-Fast Llama 3.3)',
        base_url: 'https://api.groq.com/openai/v1',
        model: 'llama-3.3-70b-versatile',
        models: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'deepseek-r1-distill-llama-70b']
      },
      openrouter: {
        provider: 'openrouter',
        name: 'OpenRouter (Universal 200+ Models)',
        base_url: 'https://openrouter.ai/api/v1',
        model: 'google/gemini-2.0-flash-exp:free',
        models: ['google/gemini-2.0-flash-exp:free', 'anthropic/claude-3.7-sonnet', 'deepseek/deepseek-r1', 'meta-llama/llama-3.3-70b-instruct:free']
      },
      custom: {
        provider: 'custom',
        name: 'Custom OpenAI-Compatible Endpoint',
        base_url: 'http://localhost:11434/v1',
        model: 'llama3',
        models: ['llama3', 'mistral', 'qwen2.5-coder', 'phi3']
      }
    };

    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const liveLatency = document.getElementById('live-latency');
    const sessionListEl = document.getElementById('session-list');
    const globalFactsListEl = document.getElementById('global-facts-list');
    const localFactsListEl = document.getElementById('local-facts-list');
    const localDrawer = document.getElementById('local-memory-drawer');

    // Auto-expand textarea
    messageInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 180) + 'px';
    });

    function handleKeyDown(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        document.getElementById('chat-form').requestSubmit();
      }
    }

    // ──────────────────────────── Settings & Providers ──────────────────────────

    async function loadSettings() {
      try {
        const resp = await fetch('/api/settings');
        if (!resp.ok) return;
        currentSettings = await resp.json();

        const prov = currentSettings.provider || 'nvidia';
        const provSelect = document.getElementById('settings-provider-select');
        if (provSelect) provSelect.value = prov;
        
        document.getElementById('settings-provider-input').value = prov;
        document.getElementById('settings-baseurl-input').value = currentSettings.base_url || 'https://integrate.api.nvidia.com/v1';
        document.getElementById('settings-model-input').value = currentSettings.model || 'meta/llama-3.1-8b-instruct';

        const badge = document.getElementById('key-configured-badge');
        if (currentSettings.is_configured) {
          badge.textContent = `✓ Saved (${currentSettings.masked_key})`;
          badge.className = 'text-[10px] text-emerald-400 font-mono';
        } else {
          badge.textContent = '✗ No key configured';
          badge.className = 'text-[10px] text-amber-400 font-mono';
        }

        renderModelSelectOptions(prov, currentSettings.model);
        updateHeaderAndFooterConfig(currentSettings);
      } catch (err) {
        console.error('Failed to load settings:', err);
      }
    }

    function handleProviderSelectChange(providerKey) {
      const p = PRESETS[providerKey] || PRESETS.nvidia;
      document.getElementById('settings-provider-input').value = p.provider;
      document.getElementById('settings-baseurl-input').value = p.base_url;
      document.getElementById('settings-model-input').value = p.model;
      renderModelSelectOptions(providerKey, p.model);
    }

    function renderModelSelectOptions(providerKey, activeModel) {
      const selectEl = document.getElementById('settings-model-select');
      const p = PRESETS[providerKey] || PRESETS.nvidia;
      const models = p.models || [];

      selectEl.innerHTML = `
        <option value="">-- Select Recommended Model --</option>
        ${models.map(m => `<option value="${m}" ${m === activeModel ? 'selected' : ''}>${m}</option>`).join('')}
      `;
    }

    function handleModelSelectChange(selectedModel) {
      if (selectedModel) {
        document.getElementById('settings-model-input').value = selectedModel;
      }
    }

    function handleGlobalPresetSelect(selectEl) {
      const val = selectEl.value;
      if (!val) return;
      const [k, v] = val.split('|');
      document.getElementById('global-key-input').value = k;
      document.getElementById('global-val-input').value = v;
      selectEl.value = '';
    }

    function toggleKeyVisibility() {
      const input = document.getElementById('settings-key-input');
      const icon = document.getElementById('key-visibility-icon');
      if (input.type === 'password') {
        input.type = 'text';
        icon.className = 'fa-solid fa-eye-slash';
      } else {
        input.type = 'password';
        icon.className = 'fa-solid fa-eye';
      }
    }

    async function handleTestConnection() {
      const btn = document.getElementById('btn-test-connection');
      const resultBox = document.getElementById('test-result-box');
      const provider = document.getElementById('settings-provider-input').value;
      const base_url = document.getElementById('settings-baseurl-input').value.trim();
      const model = document.getElementById('settings-model-input').value.trim();
      const api_key = document.getElementById('settings-key-input').value.trim();

      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin text-cyan-400 text-xs"></i> <span>Testing Endpoint...</span>';
      resultBox.classList.add('hidden');

      try {
        const resp = await fetch('/api/settings/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider, base_url, model, api_key })
        });
        const data = await resp.json();

        resultBox.classList.remove('hidden');
        if (data.success) {
          resultBox.className = 'p-2.5 rounded-lg text-xs border bg-emerald-950/40 border-emerald-500/40 text-emerald-300';
          resultBox.innerHTML = `<i class="fa-solid fa-circle-check text-emerald-400 mr-1"></i> ${escapeHtml(data.message)}`;
        } else {
          resultBox.className = 'p-2.5 rounded-lg text-xs border bg-red-950/40 border-red-500/40 text-red-300';
          resultBox.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-red-400 mr-1"></i> ${escapeHtml(data.error || 'Connection probe failed.')}`;
        }
      } catch (err) {
        resultBox.classList.remove('hidden');
        resultBox.className = 'p-2.5 rounded-lg text-xs border bg-red-950/40 border-red-500/40 text-red-300';
        resultBox.innerHTML = `❌ Network error testing connection: ${escapeHtml(err.message)}`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-plug text-xs text-cyan-400"></i> <span>Test Connection</span>';
      }
    }

    async function handleSaveSettings(e) {
      e.preventDefault();
      const provider = document.getElementById('settings-provider-input').value;
      const base_url = document.getElementById('settings-baseurl-input').value.trim();
      const model = document.getElementById('settings-model-input').value.trim();
      const api_key = document.getElementById('settings-key-input').value.trim();

      try {
        const resp = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider, base_url, model, api_key: api_key || null })
        });
        const data = await resp.json();
        if (resp.ok) {
          document.getElementById('settings-key-input').value = '';
          alert('✓ Settings encrypted & saved to local vault!');
          await loadSettings();
          if (activeSessionId) await selectSession(activeSessionId);
        } else {
          alert('Failed to save settings: ' + (data.detail || 'Unknown error'));
        }
      } catch (err) {
        alert('Network error saving settings: ' + err.message);
      }
    }

    function updateHeaderAndFooterConfig(settings) {
      if (!settings) return;
      document.getElementById('active-model-display').textContent = settings.model || 'N/A';
      document.getElementById('active-provider-display').textContent = (settings.provider || 'NVIDIA').toUpperCase();
    }

    // ──────────────────────────── Local Memory Drawer ──────────────────────────

    function toggleLocalMemoryDrawer() {
      isLocalDrawerOpen = !isLocalDrawerOpen;
      if (isLocalDrawerOpen) {
        localDrawer.classList.remove('translate-x-full');
        renderLocalFactsList();
      } else {
        localDrawer.classList.add('translate-x-full');
      }
    }

    function renderLocalFactsList() {
      const keys = Object.keys(localFactsCache);
      document.getElementById('local-count-badge').textContent = keys.length;

      if (keys.length === 0) {
        localFactsListEl.innerHTML = '<div class="text-xs text-zinc-500 italic p-3 text-center bg-zinc-950/40 rounded-xl border border-zinc-800/40">No pinned facts in this chat yet.</div>';
        return;
      }

      localFactsListEl.innerHTML = keys.map(k => {
        const val = localFactsCache[k];
        return `
          <div class="bg-zinc-950/60 border border-zinc-800/80 rounded-xl p-2.5 flex items-start justify-between group hover:border-zinc-700 transition-colors">
            <div class="min-w-0 flex-1 pr-2">
              <span class="inline-block text-[10px] font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">${escapeHtml(k)}</span>
              <p class="text-xs text-zinc-200 mt-1 leading-relaxed break-words">${escapeHtml(val)}</p>
            </div>
            <button
              onclick="deleteLocalFact('${escapeHtml(k)}')"
              class="text-zinc-500 hover:text-red-400 p-1 transition-colors"
              title="Unpin fact"
            >
              <i class="fa-solid fa-trash-can text-xs"></i>
            </button>
          </div>
        `;
      }).join('');
    }

    async function handleAddLocalFact(e) {
      e.preventDefault();
      if (!activeSessionId) return;
      const keyInput = document.getElementById('local-key-input');
      const valInput = document.getElementById('local-val-input');
      const key = keyInput.value.trim();
      const val = valInput.value.trim();
      if (!key || !val) return;

      try {
        const resp = await fetch(`/api/sessions/${activeSessionId}/local-memory`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key, value: val })
        });
        if (resp.ok) {
          const data = await resp.json();
          localFactsCache = data.facts || {};
          keyInput.value = '';
          valInput.value = '';
          renderLocalFactsList();
          updateTelemetry(data.stats);
        }
      } catch (err) {
        alert('Failed to pin local fact: ' + err.message);
      }
    }

    async function deleteLocalFact(key) {
      if (!activeSessionId) return;
      if (!confirm(`Unpin '${key}' from this conversation?`)) return;
      try {
        const resp = await fetch(`/api/sessions/${activeSessionId}/local-memory/${encodeURIComponent(key)}`, { method: 'DELETE' });
        if (resp.ok) {
          const data = await resp.json();
          localFactsCache = data.facts || {};
          renderLocalFactsList();
          updateTelemetry(data.stats);
        }
      } catch (err) {
        alert('Failed to delete local fact: ' + err.message);
      }
    }

    // ──────────────────────────── Tab Switching ──────────────────────────

    function switchSidebarTab(tab) {
      const chatsTabBtn = document.getElementById('tab-btn-chats');
      const globalTabBtn = document.getElementById('tab-btn-global');
      const settingsTabBtn = document.getElementById('tab-btn-settings');
      const chatsContent = document.getElementById('tab-content-chats');
      const globalContent = document.getElementById('tab-content-global');
      const settingsContent = document.getElementById('tab-content-settings');

      chatsTabBtn.className = 'py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 text-zinc-400 hover:text-zinc-200';
      globalTabBtn.className = 'py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 text-zinc-400 hover:text-zinc-200';
      settingsTabBtn.className = 'py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 text-zinc-400 hover:text-zinc-200';

      chatsContent.classList.add('hidden');
      globalContent.classList.add('hidden');
      settingsContent.classList.add('hidden');

      if (tab === 'chats') {
        chatsTabBtn.className = 'py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 bg-zinc-800 text-white shadow-sm';
        chatsContent.classList.remove('hidden');
      } else if (tab === 'global') {
        globalTabBtn.className = 'py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 bg-zinc-800 text-white shadow-sm';
        globalContent.classList.remove('hidden');
        loadGlobalMemory();
      } else if (tab === 'settings') {
        settingsTabBtn.className = 'py-1.5 px-2 rounded-lg transition-all flex items-center justify-center gap-1 bg-zinc-800 text-white shadow-sm';
        settingsContent.classList.remove('hidden');
        loadSettings();
      }
    }

    // ──────────────────────────── Global Memory Management ──────────────────────────

    async function loadGlobalMemory() {
      try {
        const resp = await fetch('/api/global-memory');
        if (!resp.ok) return;
        globalFactsCache = await resp.json();
        renderGlobalFactsList();
      } catch (err) {
        console.error('Failed to load global memory:', err);
      }
    }

    function renderGlobalFactsList() {
      document.getElementById('stat-global-count').textContent = globalFactsCache.length;

      if (globalFactsCache.length === 0) {
        globalFactsListEl.innerHTML = '<div class="text-xs text-zinc-500 italic p-3 text-center bg-zinc-950/40 rounded-xl border border-zinc-800/40">No global facts yet. Add one above!</div>';
        return;
      }

      globalFactsListEl.innerHTML = globalFactsCache.map(f => {
        return `
          <div class="bg-zinc-950/60 border border-zinc-800/80 rounded-xl p-2.5 flex items-start justify-between group hover:border-zinc-700 transition-colors">
            <div class="min-w-0 flex-1 pr-2">
              <span class="inline-block text-[10px] font-semibold uppercase tracking-wider text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">${escapeHtml(f.key)}</span>
              <p class="text-xs text-zinc-200 mt-1 leading-relaxed break-words">${escapeHtml(f.value)}</p>
            </div>
            <button
              onclick="deleteGlobalFact('${escapeHtml(f.key)}')"
              class="text-zinc-500 hover:text-red-400 p-1 transition-colors"
              title="Delete global fact"
            >
              <i class="fa-solid fa-trash-can text-xs"></i>
            </button>
          </div>
        `;
      }).join('');
    }

    async function handleAddGlobalFact(e) {
      e.preventDefault();
      const keyInput = document.getElementById('global-key-input');
      const valInput = document.getElementById('global-val-input');
      const key = keyInput.value.trim();
      const val = valInput.value.trim();
      if (!key || !val) return;

      try {
        const resp = await fetch('/api/global-memory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key, value: val })
        });
        if (resp.ok) {
          keyInput.value = '';
          valInput.value = '';
          await loadGlobalMemory();
          if (activeSessionId) {
            await refreshActiveSessionStats();
          }
        }
      } catch (err) {
        alert('Failed to save global fact: ' + err.message);
      }
    }

    async function deleteGlobalFact(key) {
      if (!confirm(`Remove global fact '${key}'?`)) return;
      try {
        const resp = await fetch(`/api/global-memory/${encodeURIComponent(key)}`, { method: 'DELETE' });
        if (resp.ok) {
          await loadGlobalMemory();
          if (activeSessionId) {
            await refreshActiveSessionStats();
          }
        }
      } catch (err) {
        alert('Failed to delete global fact: ' + err.message);
      }
    }

    // ──────────────────────────── Session Management ──────────────────────────

    async function loadSessions() {
      try {
        const resp = await fetch('/api/sessions');
        if (!resp.ok) return;
        sessionsCache = await resp.json();
        renderSessionList();

        const exists = sessionsCache.some(s => s.session_id === activeSessionId);
        if (activeSessionId && exists) {
          await selectSession(activeSessionId);
        } else if (sessionsCache.length > 0) {
          await selectSession(sessionsCache[0].session_id);
        } else {
          await createNewSession();
        }
      } catch (err) {
        console.error('Failed to load sessions:', err);
      }
    }

    function renderSessionList() {
      document.getElementById('total-sessions-badge').textContent = sessionsCache.length;
      if (sessionsCache.length === 0) {
        sessionListEl.innerHTML = '<div class="text-xs text-zinc-500 italic p-3 text-center">No saved chats yet.</div>';
        return;
      }

      sessionListEl.innerHTML = sessionsCache.map(s => {
        const isActive = s.session_id === activeSessionId;
        const rawTitle = s.title || 'New Conversation';
        const displayTitle = rawTitle.length > 32 ? rawTitle.substring(0, 32) + '...' : rawTitle;
        const msgCount = s.message_count || 0;
        const timeStr = s.updated_at ? new Date(s.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

        return `
          <div
            onclick="selectSession('${s.session_id}')"
            class="group p-2.5 rounded-xl cursor-pointer transition-all flex items-center justify-between border ${
              isActive
                ? 'bg-zinc-800/90 border-cyan-500/40 text-white shadow-sm'
                : 'bg-zinc-950/40 border-zinc-800/40 text-zinc-400 hover:bg-zinc-850 hover:text-zinc-200'
            }"
          >
            <div class="flex items-center space-x-2.5 min-w-0 flex-1">
              <i class="fa-regular fa-message text-xs ${isActive ? 'text-cyan-400' : 'text-zinc-500'}"></i>
              <div class="min-w-0 flex-1">
                <p class="text-xs font-medium truncate ${isActive ? 'text-zinc-100 font-semibold' : ''}">${escapeHtml(displayTitle)}</p>
                <p class="text-[10px] text-zinc-500 flex items-center gap-1.5 mt-0.5">
                  <span>${timeStr}</span>
                  <span>·</span>
                  <span>${msgCount} turns</span>
                </p>
              </div>
            </div>
            <button
              onclick="event.stopPropagation(); deleteSession('${s.session_id}')"
              class="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 p-1.5 rounded transition-all ml-1"
              title="Delete conversation"
            >
              <i class="fa-solid fa-trash-can text-xs"></i>
            </button>
          </div>
        `;
      }).join('');
    }

    async function selectSession(sessionId) {
      activeSessionId = sessionId;
      localStorage.setItem('stateful_active_session_id', sessionId);
      renderSessionList();

      document.getElementById('session-id-display').textContent = 'ID: ' + sessionId;

      try {
        const resp = await fetch(`/api/sessions/${sessionId}`);
        if (!resp.ok) {
          chatMessages.innerHTML = `<div class="p-4 text-xs text-red-400">Failed to load session details.</div>`;
          return;
        }

        const data = await resp.json();
        localFactsCache = data.local_facts || {};
        renderLocalFactsList();
        updateTelemetry(data.stats);

        // Update header
        const title = (data.messages && data.messages.length > 0) ? data.messages[0].content : 'New Conversation';
        document.getElementById('active-chat-title').textContent = title.length > 40 ? title.substring(0, 40) + '...' : title;
        document.getElementById('active-chat-meta').textContent = `${data.messages ? data.messages.length : 0} messages · Model: ${data.session.model || 'N/A'}`;

        // Render full persisted message history
        chatMessages.innerHTML = '';
        if (data.messages && data.messages.length > 0) {
          data.messages.forEach(msg => {
            appendMessage(msg.role, msg.content, null, null, false);
          });
        } else {
          chatMessages.innerHTML = `
            <div class="flex items-start space-x-3 max-w-3xl">
              <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shrink-0 mt-0.5 shadow-md shadow-cyan-600/20">
                <i class="fa-solid fa-robot text-sm"></i>
              </div>
              <div class="bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm px-4 py-3.5 text-sm text-zinc-200 space-y-2 shadow-sm">
                <p class="font-semibold text-cyan-400">New Conversation Started</p>
                <p class="text-zinc-300">This conversation is isolated with its own SQLite history and priority memory. Use the <b class="text-emerald-400">Local Memory</b> button up top to pin facts to this conversation.</p>
              </div>
            </div>
          `;
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;
      } catch (err) {
        console.error('Error selecting session:', err);
      }
    }

    async function createNewSession() {
      try {
        const resp = await fetch('/api/sessions', { method: 'POST' });
        if (!resp.ok) return;
        const newSession = await resp.json();
        await loadSessions();
        await selectSession(newSession.session_id);
      } catch (err) {
        alert('Failed to create new session: ' + err.message);
      }
    }

    async function deleteSession(sessionId) {
      if (!confirm('Are you sure you want to permanently delete this conversation?')) return;
      try {
        const resp = await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        if (resp.ok) {
          if (activeSessionId === sessionId) {
            activeSessionId = null;
            localStorage.removeItem('stateful_active_session_id');
          }
          await loadSessions();
        }
      } catch (err) {
        alert('Failed to delete session: ' + err.message);
      }
    }

    async function deleteActiveSession() {
      if (activeSessionId) {
        await deleteSession(activeSessionId);
      }
    }

    async function refreshActiveSessionStats() {
      if (!activeSessionId) return;
      try {
        const resp = await fetch(`/api/sessions/${activeSessionId}`);
        if (resp.ok) {
          const data = await resp.json();
          localFactsCache = data.local_facts || {};
          renderLocalFactsList();
          updateTelemetry(data.stats);
        }
      } catch (err) {
        console.error('Failed to refresh stats:', err);
      }
    }

    // ──────────────────────────── Chat Operations ──────────────────────────

    async function handleSend(e) {
      e.preventDefault();
      const message = messageInput.value.trim();
      if (!message) return;

      const welcome = document.getElementById('welcome-message');
      if (welcome) welcome.remove();

      appendMessage('user', message);
      messageInput.value = '';
      messageInput.style.height = 'auto';
      sendBtn.disabled = true;

      const startTime = performance.now();
      liveLatency.textContent = 'Thinking...';

      const typingId = appendTypingIndicator();

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: message, session_id: activeSessionId })
        });

        const data = await response.json();
        removeTypingIndicator(typingId);

        if (!response.ok) {
          appendMessage('system', '❌ Error: ' + (data.detail || 'Failed to process request. Check API Settings in the sidebar.'));
        } else {
          if (data.type === 'command') {
            appendMessage('command', data.content);
          } else {
            appendMessage('assistant', data.content, data.latency_sec, data.usage);
          }

          if (data.session_id && data.session_id !== activeSessionId) {
            activeSessionId = data.session_id;
            localStorage.setItem('stateful_active_session_id', activeSessionId);
          }

          if (data.local_facts) {
            localFactsCache = data.local_facts;
            renderLocalFactsList();
          }

          updateTelemetry(data.stats);
          const sessionsResp = await fetch('/api/sessions');
          if (sessionsResp.ok) {
            sessionsCache = await sessionsResp.json();
            renderSessionList();
          }
        }

        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        liveLatency.textContent = `Latency: ${elapsed}s`;
      } catch (err) {
        removeTypingIndicator(typingId);
        appendMessage('system', '❌ Network error: ' + err.message);
        liveLatency.textContent = 'Error';
      } finally {
        sendBtn.disabled = false;
        messageInput.focus();
      }
    }

    function appendMessage(role, text, latencySec, usage, shouldScroll = true) {
      const container = document.createElement('div');
      container.className = 'flex items-start space-x-3 max-w-3xl ' + (role === 'user' ? 'ml-auto justify-end' : '');

      let iconHtml = '';
      let bubbleClasses = '';
      let renderedContent = '';

      if (role === 'user') {
        bubbleClasses = 'bg-cyan-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm shadow-sm';
        renderedContent = escapeHtml(text).replace(/\\n/g, '<br/>');
        container.innerHTML = `
          <div class="${bubbleClasses} max-w-xl">
            <p class="leading-relaxed">${renderedContent}</p>
          </div>
          <div class="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center text-cyan-400 shrink-0 mt-0.5">
            <i class="fa-solid fa-user text-xs"></i>
          </div>
        `;
      } else if (role === 'assistant') {
        bubbleClasses = 'bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-zinc-200 shadow-sm prose prose-invert max-w-none';
        renderedContent = marked.parse(text);
        
        let statsMeta = '';
        if (latencySec) {
          statsMeta = `<div class="text-[10px] text-zinc-500 font-mono mt-2 pt-2 border-t border-zinc-800/80 flex items-center gap-3">
            <span><i class="fa-regular fa-clock"></i> ${latencySec.toFixed(2)}s</span>
            ${usage ? `<span><i class="fa-solid fa-coins"></i> ${usage.prompt_tokens}+${usage.completion_tokens} tok</span>` : ''}
          </div>`;
        }

        container.innerHTML = `
          <div class="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shrink-0 mt-0.5 shadow-md shadow-cyan-600/20">
            <i class="fa-solid fa-robot text-xs"></i>
          </div>
          <div class="flex-1 bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm px-4 py-3 text-sm text-zinc-200 shadow-sm">
            <div class="prose">${renderedContent}</div>
            ${statsMeta}
          </div>
        `;
      } else if (role === 'command') {
        container.innerHTML = `
          <div class="w-8 h-8 rounded-lg bg-purple-600/20 border border-purple-500/30 flex items-center justify-center text-purple-400 shrink-0 mt-0.5">
            <i class="fa-solid fa-terminal text-xs"></i>
          </div>
          <div class="flex-1 bg-zinc-900/90 border border-purple-500/20 rounded-2xl rounded-tl-sm px-4 py-3 text-xs text-purple-200 font-mono whitespace-pre-wrap shadow-sm">
${escapeHtml(text)}
          </div>
        `;
      } else {
        container.innerHTML = `
          <div class="w-full bg-red-950/40 border border-red-800/40 rounded-xl p-3 text-xs text-red-300">
            ${escapeHtml(text)}
          </div>
        `;
      }

      chatMessages.appendChild(container);
      if (shouldScroll) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }
    }

    function appendTypingIndicator() {
      const id = 'typing-' + Date.now();
      const container = document.createElement('div');
      container.id = id;
      container.className = 'flex items-start space-x-3 max-w-3xl';
      container.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-400 shrink-0 mt-0.5">
          <i class="fa-solid fa-robot text-xs"></i>
        </div>
        <div class="bg-zinc-900 border border-zinc-800 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center space-x-1.5">
          <span class="w-2 h-2 rounded-full bg-cyan-400 animate-bounce"></span>
          <span class="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style="animation-delay: 0.15s"></span>
          <span class="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style="animation-delay: 0.3s"></span>
        </div>
      `;
      chatMessages.appendChild(container);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      return id;
    }

    function removeTypingIndicator(id) {
      const el = document.getElementById(id);
      if (el) el.remove();
    }

    function updateTelemetry(stats) {
      if (!stats) return;
      document.getElementById('stat-turns').textContent = stats.active_turns || 0;
      document.getElementById('session-id-display').textContent = 'ID: ' + (stats.session_id || 'N/A');
      document.getElementById('active-model-display').textContent = stats.model || currentSettings.model || 'N/A';
      
      const current = stats.estimated_context_tokens || 0;
      const budget = stats.budget_tokens || 8192;
      document.getElementById('token-ratio').textContent = `${current.toLocaleString()} / ${budget.toLocaleString()}`;
      
      const pct = Math.min(100, Math.max(2, (current / budget) * 100));
      document.getElementById('token-bar').style.width = pct + '%';
      
      document.getElementById('stat-global-count').textContent = stats.global_facts_count || globalFactsCache.length || 0;
    }

    async function runQuickCommand(cmd) {
      messageInput.value = cmd;
      document.getElementById('chat-form').requestSubmit();
    }

    function escapeHtml(str) {
      if (!str) return '';
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    // Startup Initialization
    loadSettings();
    loadGlobalMemory();
    loadSessions();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTMLResponse(content=HTML_CONTENT)


# ──────────────────────────── Server Launch Helper ──────────────────────────

def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    print(f"[*] Starting Multi-Session Stateful Chatbot Web Server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
