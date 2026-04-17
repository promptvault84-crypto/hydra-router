import os
import json
import time
import asyncio
import aiohttp
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from aiohttp import web

# ============================================================
# SOURCES – All your connected AI model providers
# ============================================================
SOURCES = {
    "cerebras": {
        "name": "Cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "api_key": os.environ.get("CEREBRAS_API_KEY", ""),
        "model": "llama3.1-70b",
        "speed_tier": 1,
        "uncensored": False,
        "rate_limit": {"requests": 30, "window": 60},
        "daily_limit": 1000,
    },
    "groq": {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "model": "llama-3.1-70b-versatile",
        "speed_tier": 1,
        "uncensored": False,
        "rate_limit": {"requests": 30, "window": 60},
        "daily_limit": 14400,
    },
    "sambanova": {
        "name": "SambaNova",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "api_key": os.environ.get("SAMBANOVA_API_KEY", ""),
        "model": "Meta-Llama-3.1-70B-Instruct",
        "speed_tier": 2,
        "uncensored": False,
        "rate_limit": {"requests": 20, "window": 60},
        "daily_limit": 5000,
    },
    "together": {
        "name": "Together",
        "url": "https://api.together.xyz/v1/chat/completions",
        "api_key": os.environ.get("TOGETHER_API_KEY", ""),
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "speed_tier": 2,
        "uncensored": False,
        "rate_limit": {"requests": 10, "window": 60},
        "daily_limit": 1000,
    },
    "openrouter": {
        "name": "OpenRouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "model": "meta-llama/llama-3.1-70b-instruct:free",
        "speed_tier": 2,
        "uncensored": False,
        "rate_limit": {"requests": 20, "window": 60},
        "daily_limit": 5000,
    },
    "hf_abliterated": {
        "name": "HF Abliterated",
        "url": "https://api-inference.huggingface.co/models/mlabonne/Llama-3.1-70B-Instruct-abliterated/v1/chat/completions",
        "api_key": os.environ.get("HF_API_KEY", ""),
        "model": "mlabonne/Llama-3.1-70B-Instruct-abliterated",
        "speed_tier": 3,
        "uncensored": True,
        "rate_limit": {"requests": 5, "window": 60},
        "daily_limit": 500,
    },
    "hf_dolphin": {
        "name": "HF Dolphin",
        "url": "https://api-inference.huggingface.co/models/cognitivecomputations/dolphin-2.9.4-llama3.1-70b/v1/chat/completions",
        "api_key": os.environ.get("HF_API_KEY", ""),
        "model": "cognitivecomputations/dolphin-2.9.4-llama3.1-70b",
        "speed_tier": 3,
        "uncensored": True,
        "rate_limit": {"requests": 5, "window": 60},
        "daily_limit": 500,
    },
    "hf_hermes": {
        "name": "HF Hermes 3",
        "url": "https://api-inference.huggingface.co/models/NousResearch/Hermes-3-Llama-3.1-70B/v1/chat/completions",
        "api_key": os.environ.get("HF_API_KEY", ""),
        "model": "NousResearch/Hermes-3-Llama-3.1-70B",
        "speed_tier": 3,
        "uncensored": True,
        "rate_limit": {"requests": 5, "window": 60},
        "daily_limit": 500,
    },
    "horde": {
        "name": "KoboldAI Horde",
        "url": "https://aihorde.net/api/v2/generate/text/async",
        "api_key": os.environ.get("HORDE_API_KEY", "0000000000"),
        "model": "",
        "speed_tier": 5,
        "uncensored": True,
        "rate_limit": {"requests": 999, "window": 60},
        "daily_limit": 999999,
    },
}

# ============================================================
# Rate Limiting – keep exactly as before
# ============================================================
@dataclass
class RateLimitTracker:
    request_times: dict = field(default_factory=lambda: defaultdict(list))
    daily_counts: dict = field(default_factory=lambda: defaultdict(int))
    daily_reset: float = field(default_factory=time.time)

    def can_use(self, source_id):
        cfg = SOURCES[source_id]
        now = time.time()
        if now - self.daily_reset > 86400:
            self.daily_counts.clear()
            self.daily_reset = now
        if self.daily_counts[source_id] >= cfg["daily_limit"]:
            return False
        window = cfg["rate_limit"]["window"]
        max_req = cfg["rate_limit"]["requests"]
        recent = [t for t in self.request_times[source_id] if now - t < window]
        self.request_times[source_id] = recent
        return len(recent) < max_req

    def record_use(self, source_id):
        self.request_times[source_id].append(time.time())
        self.daily_counts[source_id] += 1

rate_tracker = RateLimitTracker()

# ============================================================
# API calling functions – unchanged
# ============================================================
async def call_openai_compatible(source_id, messages, max_tokens=2048, temperature=0.7):
    cfg = SOURCES[source_id]
    if source_id == "horde":
        return await call_horde(messages, max_tokens, temperature)
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                cfg["url"],
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    rate_tracker.record_use(source_id)
                    content = ""
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError):
                        content = str(data)
                    return {
                        "success": True,
                        "source": cfg["name"],
                        "model": cfg["model"],
                        "content": content,
                    }
                elif resp.status == 429:
                    return {"success": False, "error": "rate_limited"}
                else:
                    error_text = await resp.text()
                    return {"success": False, "error": f"HTTP {resp.status}: {error_text[:200]}"}
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

async def call_horde(messages, max_tokens, temperature):
    cfg = SOURCES["horde"]
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"### System:\n{content}\n\n"
        elif role == "user":
            prompt += f"### User:\n{content}\n\n"
        elif role == "assistant":
            prompt += f"### Assistant:\n{content}\n\n"
    prompt += "### Assistant:\n"
    headers = {
        "apikey": cfg["api_key"],
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "params": {
            "max_length": min(max_tokens, 512),
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
        },
        "models": [],
        "trusted_workers": False,
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://aihorde.net/api/v2/generate/text/async",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status != 202:
                    error_text = await resp.text()
                    return {"success": False, "error": f"Horde submit failed: {error_text[:200]}"}
                data = await resp.json()
                task_id = data.get("id", "")
                for attempt in range(60):
                    await asyncio.sleep(5)
                    async with session.get(
                        f"https://aihorde.net/api/v2/generate/text/status/{task_id}",
                        headers=headers,
                    ) as resp:
                        data = await resp.json()
                        if data.get("done"):
                            generations = data.get("generations", [])
                            if generations:
                                rate_tracker.record_use("horde")
                                return {
                                    "success": True,
                                    "source": "KoboldAI Horde",
                                    "model": generations[0].get("model", "unknown"),
                                    "content": generations[0].get("text", ""),
                                }
                return {"success": False, "error": "Horde timeout after 5 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}

# ============================================================
# NEW: Manual model selection handler – NO auto‑routing
# ============================================================
async def handle_chat(request):
    data = await request.json()
    messages = data.get("messages", [])
    model_id = data.get("model_id")
    max_tokens = data.get("max_tokens", 2048)
    temperature = data.get("temperature", 0.7)

    # 1. Must provide a model_id
    if not model_id:
        return web.json_response({
            "success": False,
            "error": "No model selected. Please choose a model."
        })

    # 2. Model must exist in SOURCES
    if model_id not in SOURCES:
        return web.json_response({
            "success": False,
            "error": f"Unknown model: {model_id}"
        })

    cfg = SOURCES[model_id]
    if not cfg["api_key"]:
        return web.json_response({
            "success": False,
            "error": f"Model {cfg['name']} is not configured (missing API key)."
        })

    # 3. Try up to 3 times (only for rate‑limit retries)
    max_attempts = 3
    for attempt in range(max_attempts):
        if not rate_tracker.can_use(model_id):
            await asyncio.sleep(2)
            continue

        result = await call_openai_compatible(model_id, messages, max_tokens, temperature)
        if result.get("success"):
            return web.json_response(result)

        if result.get("error") == "rate_limited":
            # Artificially exhaust the rate limit so can_use stays false
            for _ in range(50):
                rate_tracker.record_use(model_id)
            await asyncio.sleep(2)
            continue
        else:
            # Other error (timeout, HTTP error) – fail immediately
            return web.json_response(result)

    return web.json_response({
        "success": False,
        "error": f"Model {cfg['name']} is currently rate‑limited. Try again in a minute."
    })

# ============================================================
# Static frontend serving
# ============================================================
async def handle_index(request):
    return web.FileResponse('./static/index.html')

async def handle_status(request):
    status = {}
    for sid, cfg in SOURCES.items():
        if not cfg["api_key"]:
            status[cfg["name"]] = "not configured"
        elif rate_tracker.can_use(sid):
            status[cfg["name"]] = "available"
        else:
            status[cfg["name"]] = "rate-limited"
    return web.json_response(status)

def create_app():
    app = web.Application()
    app.router.add_static('/static/', path='./static', name='static')
    app.router.add_get("/", handle_index)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/status", handle_status)
    return app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app = create_app()
    print(f"FACEBONY (manual control) starting on port {port}")
    configured = [v["name"] for k, v in SOURCES.items() if v["api_key"]]
    print(f"Configured backends: {configured}")
    web.run_app(app, host="0.0.0.0", port=port)
