import time, json, urllib.request, urllib.error

_target = {"base_url":"http://127.0.0.1:11434","model":"llama3:latest"}
_last_call = 0.0
_min_interval = 6.0
_burst = 1

def set_llm_target(base_url, model):
    _target["base_url"] = base_url
    _target["model"] = model

def llm_rate_limits(min_interval_sec=6.0, burst=1):
    """Simple throttle to avoid hammering local LLMs."""
    global _min_interval, _burst
    _min_interval = float(min_interval_sec)
    _burst = int(burst)

def _ok_to_call():
    global _last_call
    now = time.time()
    if now - _last_call >= _min_interval:
        _last_call = now
        return True
    return False

def get_models(base_url):
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=4) as r:
            data = json.loads(r.read().decode("utf-8"))
            names = []
            for it in data.get("models", []):
                name = it.get("name") or it.get("model") or ""
                if name: names.append(name)
            return names
    except Exception:
        return []

def llm_status_line():
    return "Status: LLM ready"

def pack_reply(user_text: str) -> str:
    """Short Dong-ish reply for chats/prayers."""
    if not _ok_to_call():
        return "« busy; please wait »"

    prompt = (
        "You translate between a human and tiny creatures called DONGS. "
        "Answer as a friendly Dong would, under 25 words, simple, playful.\n\n"
        f"Message: {user_text}\n"
        "Dong:"
    )
    body = json.dumps({"model": _target["model"], "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(f"{_target['base_url']}/api/generate", data=body, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
            return (data.get("response") or "").strip() or "…"
    except urllib.error.URLError:
        return "« network error »"
    except Exception:
        return "« error »"
