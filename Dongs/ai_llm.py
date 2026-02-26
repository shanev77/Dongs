import time, json, urllib.request, urllib.error

_target = {
    "base_url": "http://192.168.0.202:11434",
    "model": "llama3:latest",
    "provider": "ollama",
    "cluster_hint": "",
}
_last_call = 0.0
_min_interval = 6.0
_burst = 1

def set_llm_target(base_url, model, provider="ollama", cluster_hint=""):
    _target["base_url"] = base_url
    _target["model"] = model
    _target["provider"] = provider
    _target["cluster_hint"] = cluster_hint

def llm_rate_limits(min_interval_sec=6.0, burst=1):
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
    where = _target.get("cluster_hint") or "direct"
    return f"Status: {_target['provider']} ready via {where}"

def _persona_prompt(user_text: str, life_state: dict | None = None) -> str:
    life_state = life_state or {}
    mood = life_state.get("mood", "curious")
    energy = float(life_state.get("energy", 6.0))
    hunger = float(life_state.get("hunger", 0.4))
    social = float(life_state.get("social_need", 0.5))
    population = int(life_state.get("population", 0))
    return (
        "You translate between a human and tiny creatures called DONGS. "
        "Speak as one Dong using simple playful language, under 28 words. "
        "Keep continuity with the life-state and mention only what fits naturally.\n"
        f"Life-state: mood={mood}, energy={energy:.2f}, hunger={hunger:.2f}, social_need={social:.2f}, population={population}\n\n"
        f"User: {user_text}\n"
        "Dong:"
    )


def pack_reply(user_text: str, life_state: dict | None = None) -> str:
    """Short Dong-ish reply for terminal chats."""
    if not _ok_to_call():
        return "« busy; please wait »"

    prompt = _persona_prompt(user_text, life_state)
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
