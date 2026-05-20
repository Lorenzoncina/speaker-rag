import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import requests
import torch
import numpy as np
import base64
from pyvis.network import Network
from transformers import pipeline

# Allow importing from the sibling `tests/` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.meld_pairs import build_pairs, load_pairs, DEFAULT_OUT as DEFAULT_PAIRS_PATH
from tests.vox1_pairs import (
    build_pairs as build_vox1_pairs,
    load_pairs as load_vox1_pairs,
    DEFAULT_OUT as DEFAULT_VOX1_PAIRS_PATH,
)
from tests.eval_speaker_verification import (
    run_experiment_direct_audio,
    run_experiment_rag_text,
    save_results,
    format_summary,
)

# --- Configuration ---
FASTAPI_URL = "http://127.0.0.1:8006" # fbk server (requires SSH Port Forwarding)
EVAL_RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "tests" / "eval_results")
LONGMEMEVAL_RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "tests" / "longmemeval_results")
MLCSLM_RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "tests" / "mlcslm_results")

# ==========================================
# 🧠 "Bring Your Own LLM" Initialization
# ==========================================
# Qwen2-7B is the UI-side demo answerer used when KG extraction runs against
# the Claude API. It is loaded LAZILY (on first answer call) so the container
# doesn't waste ~14 GB of VRAM at startup when the user only wants to run a
# benchmark in Local mode (where Qwen3-14B on the backend handles answering).
# When the user switches the backend to Local mode, this stays unloaded — and
# if it was already loaded, the UI mode-switch handler unloads it explicitly
# so the GPU has room for Qwen3-14B on the backend container.
_ui_llm_cache: dict = {"pipeline": None}


def get_ui_llm():
    """Lazy-load Qwen/Qwen2-7B-Instruct on first use."""
    if _ui_llm_cache["pipeline"] is None:
        print("Loading Local Text LLM for the UI (Qwen/Qwen2-7B-Instruct)...")
        _ui_llm_cache["pipeline"] = pipeline(
            "text-generation",
            model="Qwen/Qwen2-7B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("UI LLM Loaded successfully.")
    return _ui_llm_cache["pipeline"]


def unload_ui_llm() -> str:
    """Drop the Qwen2-7B pipeline and free its VRAM. Used when switching to
    Local mode so Qwen3-14B on the backend has room to load."""
    import gc
    if _ui_llm_cache["pipeline"] is None:
        return "Qwen2-7B was not loaded — nothing to free."
    try:
        pipe = _ui_llm_cache["pipeline"]
        _ui_llm_cache["pipeline"] = None
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Qwen2-7B unloaded from UI container."
    except Exception as e:
        return f"Failed to unload Qwen2-7B: {e}"

# ==========================================
# Lazy loader for Qwen2-Audio (only used by Experiment 1)
# ==========================================
_qwen_audio_cache = {"model": None, "processor": None}

def get_qwen_audio():
    if _qwen_audio_cache["model"] is None:
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        print("Lazy-loading Qwen/Qwen2-Audio-7B-Instruct ...")
        _qwen_audio_cache["processor"] = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct"
        )
        _qwen_audio_cache["model"] = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("Qwen2-Audio loaded.")
    return _qwen_audio_cache["model"], _qwen_audio_cache["processor"]

# ==========================================
# Speaker RAG API Functions
# ==========================================
def enroll_rag(audio_paths, user_ids):
    if not audio_paths:
        return "Please upload audio files to enroll."
    
    files = [('audios', (os.path.basename(path), open(path, 'rb'), 'audio/wav')) for path in audio_paths]
    
    data = {}
    if user_ids and user_ids.strip():
        data = {'user_ids': [uid.strip() for uid in user_ids.split(',')]}
        
    try:
        response = requests.post(f"{FASTAPI_URL}/rag/index", files=files, data=data)
        if response.status_code == 200:
            return f"Success:\n{response.json()}"
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def list_enrolled_users():
    try:
        response = requests.get(f"{FASTAPI_URL}/rag/enrolled")
        if response.status_code == 200:
            res_json = response.json()
            users = res_json.get("enrolled_users", {})
            
            if not users:
                return "No users are currently enrolled in the database."
            
            lines = [f"📊 Total Enrolled Users: {len(users)}\n" + "-"*45]
            for uid, attrs in users.items():
                files = ", ".join(attrs.get("audio_files", []))
                lines.append(f"👤 User ID: {uid}")
                lines.append(f"   ➤ Attributes: {attrs.get('age')} | {attrs.get('gender')} | {attrs.get('language')}")
                lines.append(f"   ➤ Source Files: {files}\n")
            return "\n".join(lines)
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def _backend_extractor_mode() -> str:
    """Return the backend's current extractor mode ('anthropic' / 'local' / 'off').

    Falls back to 'off' if the backend is unreachable so the UI doesn't hang on
    a misconfigured environment.
    """
    try:
        r = requests.get(f"{FASTAPI_URL}/admin/extractor_mode", timeout=5)
        if r.status_code == 200:
            return str(r.json().get("mode", "off")).lower()
    except Exception:
        pass
    return "off"


def _answer_with_backend_local(prompt: str, max_new_tokens: int = 300) -> str:
    """Run the fact-sheet prompt through the backend's loaded Qwen3-14B."""
    r = requests.post(
        f"{FASTAPI_URL}/llm/generate",
        json={"prompt": prompt, "max_new_tokens": int(max_new_tokens)},
        timeout=600,
    )
    if r.status_code != 200:
        raise RuntimeError(f"backend /llm/generate {r.status_code}: {r.text[:200]}")
    return (r.json().get("text") or "").strip()


def _answer_with_ui_qwen(prompt: str, max_new_tokens: int = 300) -> str:
    """Run the fact-sheet prompt through the UI-side Qwen2-7B (lazy-loaded)."""
    pipe = get_ui_llm()
    messages = [
        {"role": "system", "content": "You are a helpful and precise analytical assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(formatted, max_new_tokens=max_new_tokens, temperature=0.1)
    return outputs[0]["generated_text"][len(formatted):].strip()


def answer_with_llm(prompt: str, max_new_tokens: int = 300) -> str:
    """Route fact-sheet answering to the right model based on backend mode.

    - Backend mode == 'local': call the backend's Qwen3-14B (one model loaded).
    - Backend mode != 'local': use the UI's Qwen2-7B (decoupled answerer).
    """
    if _backend_extractor_mode() == "local":
        return _answer_with_backend_local(prompt, max_new_tokens=max_new_tokens)
    return _answer_with_ui_qwen(prompt, max_new_tokens=max_new_tokens)


def ask_rag(audio_paths, question, history):
    history = history or []

    if not question or not question.strip():
        return "Please provide a question.", "", history, _format_history(history)

    # Hybrid: audio optional. If missing, the backend extracts speaker names from the question.
    files = None
    if audio_paths:
        files = [('audios', (os.path.basename(path), open(path, 'rb'), 'audio/wav')) for path in audio_paths]

    data = {'question': question}
    if history:
        data['conversation_history'] = json.dumps(history)

    try:
        response = requests.post(f"{FASTAPI_URL}/rag/retrieve", files=files, data=data)
        if response.status_code == 200:
            res_json = response.json()

            fact_sheet = res_json.get('fact_sheet', "No context provided by the backend.")
            recommended_prompt = res_json.get('recommended_prompt', "")

            if not recommended_prompt:
                return "Error: Backend did not return a prompt.", fact_sheet, history, _format_history(history)

            final_answer = answer_with_llm(recommended_prompt, max_new_tokens=300)

            new_history = history + [{"question": question, "answer": final_answer}]
            return final_answer, fact_sheet, new_history, _format_history(new_history)

        else:
            err = f"Error {response.status_code}:\n{response.text}"
            return err, "Error fetching context.", history, _format_history(history)

    except Exception as e:
        return f"An error occurred: {str(e)}", "", history, _format_history(history)


def _format_history(history):
    if not history:
        return "(no prior turns)"
    lines = []
    for i, turn in enumerate(history, start=1):
        lines.append(f"Turn {i}")
        lines.append(f"  Q: {turn.get('question', '')}")
        lines.append(f"  A: {turn.get('answer', '')}")
    return "\n".join(lines)


def reset_history():
    return [], _format_history([])


def save_chat_to_memory(history):
    """Enroll the current chat history into the RAG memory layer.

    The conversation_state holds {question, answer} pairs; we flatten them
    into a {role, content} sequence so each side becomes its own utterance
    in the graph + semantic store.
    """
    if not history:
        return "No conversation to save yet — ask a question first."

    messages = []
    for turn in history:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})

    if not messages:
        return "No conversation to save yet — ask a question first."

    try:
        response = requests.post(
            f"{FASTAPI_URL}/rag/index_chat_session",
            json={"messages": messages},
            timeout=120,
        )
        if response.status_code == 200:
            r = response.json()
            return (
                f"✅ Saved {r.get('enrolled_count', 0)} chat turns to memory "
                f"({r.get('user_turns', 0)} user / {r.get('assistant_turns', 0)} assistant) "
                f"as dialogue {r.get('dialogue_id', '?')}."
            )
        return f"Error {response.status_code}:\n{response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def clear_rag():
    try:
        response = requests.delete(f"{FASTAPI_URL}/rag/clear")
        if response.status_code == 200:
            return "Database cleared successfully."
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def ingest_meld_ui():
    try:
        response = requests.get(f"{FASTAPI_URL}/admin/ingest_meld")
        if response.status_code == 200:
            res_json = response.json()
            return f"MELD Ingestion Complete!\nProcessed: {res_json.get('enrolled_count', 0)} files."
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


def ingest_voxceleb_ui(per_speaker: int = 10):
    # Read the probe file list from the local pairs JSON and send it in the
    # request body.  This avoids any shared-filesystem dependency between the
    # UI container and the backend container.
    probe_rel_paths = []
    if Path(DEFAULT_VOX1_PAIRS_PATH).exists():
        pairs = load_vox1_pairs(DEFAULT_VOX1_PAIRS_PATH)
        seen = set()
        for p in pairs:
            for key in ("audio1_rel", "audio2_rel"):
                v = p.get(key, "")
                if v and v not in seen:
                    seen.add(v)
                    probe_rel_paths.append(v)

    payload = {"per_speaker": int(per_speaker), "probe_rel_paths": probe_rel_paths}
    try:
        response = requests.post(
            f"{FASTAPI_URL}/admin/ingest_voxceleb",
            json=payload,
            timeout=3600,
        )
        if response.status_code == 200:
            r = response.json()
            probe_note = f"({len(probe_rel_paths)} probe files excluded)" if probe_rel_paths else "(full Vox1-O fallback)"
            return (
                f"VoxCeleb1 Ingestion Complete! {probe_note}\n"
                f"Speakers: {r.get('n_speakers', 0)}\n"
                f"Per speaker: {r.get('per_speaker', 0)}\n"
                f"Total enrolled: {r.get('enrolled_count', 0)} files."
            )
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# ==========================================
# Extractor-mode management (Claude API vs local Qwen3-14B)
# ==========================================
def _format_mode_status(status: dict) -> str:
    """Render a backend mode/status payload as a single human-readable line."""
    if not isinstance(status, dict):
        return "Unknown status."
    mode = str(status.get("mode", "off")).lower()
    enabled = bool(status.get("enabled"))
    model = status.get("model") or "unknown"
    reason = status.get("reason") or "unknown reason"
    if mode == "anthropic" and enabled:
        return f"✅ Claude API mode — KG model: {model}.  Answers: UI Qwen2-7B."
    if mode == "local" and enabled:
        return f"✅ Local Qwen3-14B mode (8-bit) — KG model: {model}.  Answers: backend Qwen3-14B."
    if mode == "anthropic" and not enabled:
        return f"⚠️ Claude API mode selected but extractor disabled — {reason}"
    if mode == "local" and not enabled:
        return f"⚠️ Local mode selected but model failed to load — {reason}"
    return f"⚠️ Extractor disabled — {reason}"


def get_mode_status_ui() -> str:
    try:
        r = requests.get(f"{FASTAPI_URL}/admin/extractor_mode", timeout=5)
        if r.status_code != 200:
            return f"Error {r.status_code}: {r.text}"
        return _format_mode_status(r.json())
    except Exception as e:
        return f"An error occurred: {e}"


def activate_claude_mode(api_key: str):
    """Switch the backend to Anthropic mode. Returns (status_line, cleared_key_input).

    Does NOT eagerly load Qwen2-7B here — that happens on the first answer
    request via `get_ui_llm()`, keeping the mode switch instant.
    """
    if not api_key or not api_key.strip():
        return "Please paste an Anthropic API key first.", ""
    try:
        r = requests.post(
            f"{FASTAPI_URL}/admin/extractor_mode",
            json={"mode": "anthropic", "api_key": api_key.strip()},
            timeout=30,
        )
        if r.status_code == 200:
            return _format_mode_status(r.json()), ""
        try:
            return _format_mode_status(r.json()), api_key
        except Exception:
            return f"Error {r.status_code}: {r.text}", api_key
    except Exception as e:
        return f"An error occurred: {e}", api_key


def activate_local_mode():
    """Switch the backend to Local mode and load Qwen3-14B eagerly.

    Before triggering the backend load we unload the UI's Qwen2-7B (if any) so
    the GPU has headroom for the 8-bit Qwen3-14B (~16 GB). The backend request
    blocks until the model is on GPU, so the user sees a single 'ready' status
    once everything is in place.
    """
    unload_msg = unload_ui_llm()
    try:
        r = requests.post(
            f"{FASTAPI_URL}/admin/extractor_mode",
            json={"mode": "local"},
            timeout=900,  # first run pulls ~28 GB from HF; subsequent loads ~30 s
        )
        if r.status_code == 200:
            return f"{_format_mode_status(r.json())}\n{unload_msg}"
        try:
            return f"{_format_mode_status(r.json())}\n{unload_msg}"
        except Exception:
            return f"Error {r.status_code}: {r.text}\n{unload_msg}"
    except Exception as e:
        return f"An error occurred: {e}\n{unload_msg}"


def deactivate_extractor():
    try:
        r = requests.post(
            f"{FASTAPI_URL}/admin/extractor_mode",
            json={"mode": "off"},
            timeout=60,
        )
        if r.status_code == 200:
            return _format_mode_status(r.json())
        return f"Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"An error occurred: {e}"


# ==========================================
# Knowledge Graph Visualization
# ==========================================
_ENTITY_COLORS = {
    # New LLM-extractor types
    "PERSON":       "#2ecc71",
    "LOCATION":     "#e74c3c",
    "ORGANIZATION": "#e67e22",
    "EVENT":        "#1abc9c",
    "OBJECT":       "#9b59b6",
    "CONCEPT":      "#3498db",
    "WORK":         "#f39c12",
    # Legacy spaCy/regex types (kept for back-compat with old graphs)
    "ORG":          "#e67e22",
    "GPE":          "#e74c3c",
    "PRODUCT":      "#9b59b6",
    "WORK_OF_ART":  "#f39c12",
    "FAC":          "#d35400",
    "UNKNOWN":      "#95a5a6",
}

_EMPTY_GRAPH_HTML = """
<div style="display:flex;align-items:center;justify-content:center;height:620px;
            background:#f8fafc;border-radius:8px;border:1px dashed #cbd5e1;">
  <div style="text-align:center;color:#64748b;">
    <div style="font-size:48px;margin-bottom:12px;">🕸️</div>
    <div style="font-size:16px;font-weight:600;">Knowledge graph is empty</div>
    <div style="font-size:13px;margin-top:6px;">Enroll some audio files first, then click Refresh.</div>
  </div>
</div>
"""

def get_graph_viz(max_entities: int = 40):
    """Fetch graph data from the backend and render an interactive pyvis/vis.js graph."""
    try:
        response = requests.get(
            f"{FASTAPI_URL}/rag/graph_viz",
            params={"max_entities": int(max_entities)},
            timeout=30,
        )
        if response.status_code != 200:
            return f"<p style='color:red'>Error {response.status_code}: {response.text}</p>", ""

        data     = response.json()
        stats    = data.get("stats", {})
        nodes    = data.get("nodes", [])
        edges    = data.get("edges", [])

        stats_str = (
            f"Nodes: {stats.get('nodes', 0)}  |  "
            f"Edges: {stats.get('edges', 0)}  |  "
            f"Speakers: {stats.get('speaker', 0)}  |  "
            f"Entities: {stats.get('entity', 0)}  |  "
            f"Utterances (hidden): {stats.get('utterance', 0)}"
        )

        if not nodes:
            return _EMPTY_GRAPH_HTML, stats_str

        # Build pyvis network
        net = Network(
            height="640px",
            width="100%",
            bgcolor="#f8fafc",
            font_color="#1e293b",
            cdn_resources="in_line",
        )

        for n in nodes:
            nid = n["id"]
            if n["type"] == "speaker":
                utt_cnt = n.get("utterance_count", 0)
                net.add_node(
                    nid,
                    label=nid.capitalize(),
                    shape="box",
                    color={"background": "#2563eb", "border": "#1d4ed8",
                           "highlight": {"background": "#60a5fa", "border": "#2563eb"}},
                    font={"color": "#ffffff", "size": 14, "bold": True},
                    size=28,
                    title=(
                        f"<b>Speaker: {nid.capitalize()}</b><br>"
                        f"Utterances enrolled: {utt_cnt}"
                    ),
                    borderWidth=2,
                )
            else:
                ent_label   = n.get("label", "UNKNOWN")
                surface     = n.get("surface", nid)
                total_ment  = n.get("total_mentions", 1)
                description = n.get("description", "")
                color       = _ENTITY_COLORS.get(ent_label, "#95a5a6")
                node_size   = min(10 + total_ment * 3, 30)
                tooltip = (
                    f"<b>{surface}</b><br>"
                    f"Type: {ent_label}<br>"
                    f"Total mentions: {total_ment}"
                )
                if description:
                    tooltip += f"<br><i>{description[:300]}</i>"
                net.add_node(
                    nid,
                    label=surface[:24],
                    shape="dot",
                    color={"background": color, "border": color,
                           "highlight": {"background": color, "border": "#1e293b"}},
                    size=node_size,
                    title=tooltip,
                    font={"size": 11},
                )

        for e in edges:
            if e.get("is_typed_fact"):
                rel = e.get("relation", "FACT")
                fact = e.get("fact", "")
                src_utts = e.get("source_utterances", [])
                tooltip = f"<b>{rel}</b>"
                if fact:
                    tooltip += f"<br>{fact}"
                if src_utts:
                    tooltip += f"<br><i>Supported by {len(src_utts)} utterance(s)</i>"
                net.add_edge(
                    e["source"], e["target"],
                    label=rel,
                    color={"color": "#7c3aed", "highlight": "#a78bfa"},
                    font={"size": 10, "color": "#5b21b6", "strokeWidth": 2, "strokeColor": "#ffffff", "bold": True},
                    width=2.5,
                    title=tooltip,
                    arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
                    dashes=False,
                )
            elif e["relation"] == "KNOWS":
                n_dial = e.get("n_dialogues", 0)
                net.add_edge(
                    e["source"], e["target"],
                    label=f"KNOWS ({n_dial})",
                    color={"color": "#1e3a5f", "highlight": "#2563eb"},
                    font={"size": 10, "color": "#1e3a5f", "strokeWidth": 2, "strokeColor": "#ffffff"},
                    width=2 + min(n_dial, 5),
                    title=f"KNOWS — {n_dial} shared dialogue(s)",
                    dashes=False,
                )
            else:
                cnt = e.get("count", 1)
                net.add_edge(
                    e["source"], e["target"],
                    label=f"MENTIONS ({cnt}×)",
                    color={"color": "#94a3b8", "highlight": "#64748b"},
                    font={"size": 9, "color": "#64748b", "strokeWidth": 2, "strokeColor": "#ffffff"},
                    width=max(0.5, min(cnt * 0.6, 3.0)),
                    title=f"MENTIONS — {cnt}×",
                    dashes=True,
                )

        net.set_options("""{
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -60,
              "springLength": 180,
              "springConstant": 0.08,
              "damping": 0.4
            },
            "stabilization": {"iterations": 200}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "dragNodes": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": false}},
            "font": {"align": "middle"}
          }
        }""")

        html_str = net.generate_html()
        b64      = base64.b64encode(html_str.encode()).decode()
        iframe   = (
            f'<iframe src="data:text/html;base64,{b64}" '
            f'width="100%" height="650px" '
            f'style="border:none;border-radius:8px;"></iframe>'
        )
        return iframe, stats_str

    except Exception as e:
        return f"<p style='color:red'>Failed to render graph: {e}</p>", ""


# ==========================================
# Speaker Verification Evaluation (MELD dev)
# ==========================================
def _ensure_pairs(regenerate: bool):
    if regenerate or not Path(DEFAULT_PAIRS_PATH).exists():
        build_pairs(output_path=DEFAULT_PAIRS_PATH)
    return load_pairs(DEFAULT_PAIRS_PATH)


def _slice_pairs(pairs, n):
    n = int(n)
    return pairs if n <= 0 else pairs[:n]


def run_exp1(n_pairs, regenerate, progress=gr.Progress()):
    try:
        pairs = _slice_pairs(_ensure_pairs(regenerate), n_pairs)
        progress(0, desc=f"Loading Qwen2-Audio (first run only)...")
        model, processor = get_qwen_audio()

        def cb(done, total):
            progress(done / max(total, 1), desc=f"Exp 1: {done}/{total} pairs")

        scores, labels = run_experiment_direct_audio(pairs, model, processor, progress_callback=cb)
        path, _ = save_results(EVAL_RESULTS_DIR, "1", scores, labels,
                               "Qwen/Qwen2-Audio-7B-Instruct")
        return format_summary("1", "Qwen/Qwen2-Audio-7B-Instruct", scores, labels, path)
    except Exception as e:
        return f"Experiment 1 failed: {e}"


def _run_rag_text_exp(experiment_id: str, label: str, n_pairs, regenerate, progress=gr.Progress()):
    try:
        pairs = _slice_pairs(_ensure_pairs(regenerate), n_pairs)

        def cb(done, total):
            progress(done / max(total, 1), desc=f"Exp {experiment_id}: {done}/{total} pairs")

        scores, labels = run_experiment_rag_text(
            pairs, get_ui_llm(), FASTAPI_URL, progress_callback=cb
        )
        extra = {"rag": "no_enrollment" if experiment_id == "2" else "meld_train_enrolled"}
        path, _ = save_results(EVAL_RESULTS_DIR, experiment_id, scores, labels,
                               "Qwen/Qwen2-7B-Instruct", extra=extra)
        return format_summary(experiment_id, label, scores, labels, path)
    except Exception as e:
        return f"Experiment {experiment_id} failed: {e}"


def run_exp2(n_pairs, regenerate, progress=gr.Progress()):
    return _run_rag_text_exp("2", "Qwen/Qwen2-7B-Instruct + RAG (empty DB)",
                             n_pairs, regenerate, progress)


def run_exp3(n_pairs, regenerate, progress=gr.Progress()):
    return _run_rag_text_exp("3", "Qwen/Qwen2-7B-Instruct + RAG (MELD train enrolled)",
                             n_pairs, regenerate, progress)


# ==========================================
# Speaker Verification Evaluation (VoxCeleb1 Vox1-O)
# ==========================================
def _ensure_vox1_pairs(regenerate: bool, n_pairs: int):
    n = int(n_pairs)
    if regenerate or not Path(DEFAULT_VOX1_PAIRS_PATH).exists():
        build_vox1_pairs(n_pairs=n, output_path=DEFAULT_VOX1_PAIRS_PATH)
    return load_vox1_pairs(DEFAULT_VOX1_PAIRS_PATH)


def build_vox1_pairs_ui(n_pairs):
    pairs = _ensure_vox1_pairs(regenerate=True, n_pairs=n_pairs)
    n = int(n_pairs)
    n_same = sum(1 for p in pairs if p["label"] == 1)
    n_diff = sum(1 for p in pairs if p["label"] == 0)
    probe_count = len({p["audio1_rel"] for p in pairs} | {p["audio2_rel"] for p in pairs})
    label = "all" if n <= 0 else str(n)
    return (
        f"Built pairs file: {label} pairs ({n_same} same, {n_diff} diff)\n"
        f"Probe files to exclude from enrollment: {probe_count}\n"
        f"Saved to: {DEFAULT_VOX1_PAIRS_PATH}"
    )


def run_vox1_exp1(n_pairs, regenerate, progress=gr.Progress()):
    try:
        pairs = _slice_pairs(_ensure_vox1_pairs(regenerate, n_pairs), n_pairs)
        progress(0, desc="Loading Qwen2-Audio (first run only)...")
        model, processor = get_qwen_audio()

        def cb(done, total):
            progress(done / max(total, 1), desc=f"Vox1 Exp 1: {done}/{total} pairs")

        scores, labels = run_experiment_direct_audio(pairs, model, processor, progress_callback=cb)
        path, _ = save_results(EVAL_RESULTS_DIR, "1", scores, labels,
                               "Qwen/Qwen2-Audio-7B-Instruct",
                               extra={"dataset": "vox1"})
        return format_summary("1 (Vox1)", "Qwen/Qwen2-Audio-7B-Instruct",
                              scores, labels, path)
    except Exception as e:
        return f"Vox1 Experiment 1 failed: {e}"


def _run_vox1_rag_text_exp(experiment_id: str, label: str, rag_tag: str,
                            n_pairs, regenerate, progress=gr.Progress()):
    try:
        pairs = _slice_pairs(_ensure_vox1_pairs(regenerate, n_pairs), n_pairs)

        def cb(done, total):
            progress(done / max(total, 1),
                     desc=f"Vox1 Exp {experiment_id}: {done}/{total} pairs")

        scores, labels = run_experiment_rag_text(
            pairs, get_ui_llm(), FASTAPI_URL, progress_callback=cb
        )
        path, _ = save_results(EVAL_RESULTS_DIR, experiment_id, scores, labels,
                               "Qwen/Qwen2-7B-Instruct",
                               extra={"dataset": "vox1", "rag": rag_tag})
        return format_summary(f"{experiment_id} (Vox1)", label, scores, labels, path)
    except Exception as e:
        return f"Vox1 Experiment {experiment_id} failed: {e}"


def run_vox1_exp2(n_pairs, regenerate, progress=gr.Progress()):
    return _run_vox1_rag_text_exp(
        "2", "Qwen/Qwen2-7B-Instruct + RAG (empty DB)", "no_enrollment",
        n_pairs, regenerate, progress,
    )


def run_vox1_exp3(n_pairs, regenerate, progress=gr.Progress()):
    return _run_vox1_rag_text_exp(
        "3", "Qwen/Qwen2-7B-Instruct + RAG (VoxCeleb1 enrolled)", "vox1_enrolled",
        n_pairs, regenerate, progress,
    )


# ==========================================
# LongMemEval benchmark
# ==========================================
def _qwen_answer(prompt: str, max_new_tokens: int = 300) -> str:
    """Run the active answerer on a single prompt.

    Delegates to `answer_with_llm` so LongMemEval runs respect the chosen mode:
    in Local mode every prompt is dispatched to the backend's Qwen3-14B; in
    Claude mode the UI's Qwen2-7B handles it.
    """
    return answer_with_llm(prompt, max_new_tokens=max_new_tokens)


# Adapted from LongMemEval paper Appendix D (Figure 13, non-CoN prompt).
_LME_LLM_ONLY_PROMPT = """I will give you several history chats between you and a user. \
Please answer the question based on the relevant chat history.

History Chats:
{chat_history}

Current Date: {question_date}

Question: {question}

Answer:"""


def _flatten_haystack(haystack_sessions, haystack_dates, max_chars: int = 80000) -> str:
    """Flatten the haystack into a single text blob, truncating from the end if too long.

    The backend's Qwen3-14B-AWQ engine is loaded with max_model_len=32k tokens
    (~96k chars). We cap at 80k chars to leave room for the prompt scaffolding
    and answer. The full `s_cleaned` haystack averages ~115k tokens and will
    be truncated; the `oracle` split typically fits without truncation.
    """
    lines = []
    for i, session in enumerate(haystack_sessions):
        date = haystack_dates[i] if i < len(haystack_dates) else "?"
        lines.append(f"=== Session {i + 1} ({date}) ===")
        for msg in session:
            role = msg.get("role", "")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("")
    full = "\n".join(lines)
    if len(full) > max_chars:
        full = full[:max_chars] + "\n\n[...truncated due to context limit...]"
    return full


def _lme_questions(split: str, n_questions: int):
    """Fetch the question list for a split. Raises on HTTP error."""
    r = requests.get(
        f"{FASTAPI_URL}/admin/longmemeval/questions",
        params={"split": split, "limit": int(n_questions)},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch questions: {r.status_code} {r.text}")
    return r.json().get("questions", [])


def _lme_open_output(mode_tag: str, split: str):
    """Create the results directory and return (output_path, file_handle) for incremental writes."""
    os.makedirs(LONGMEMEVAL_RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(LONGMEMEVAL_RESULTS_DIR) / f"longmemeval_{mode_tag}_{split}_{stamp}.jsonl"
    return path, open(path, "w")


def run_longmemeval_rag(split, n_questions, enable_kg, progress=gr.Progress()):
    """Run the benchmark with SpeakerRAG: per-question clear + enroll + retrieve, then Qwen."""
    try:
        questions = _lme_questions(split, n_questions)
    except Exception as e:
        return f"Failed to load split: {e}"

    total = len(questions)
    if total == 0:
        return f"No questions found in split '{split}'."

    out_path, out_f = _lme_open_output("rag", split)
    log = [
        f"Running LongMemEval (RAG mode) on '{split}' — {total} questions",
        f"  KG extraction during enrollment: {'ON' if enable_kg else 'OFF'}",
        f"  Output: {out_path}",
        "",
    ]

    try:
        for i, q in enumerate(questions):
            progress(i / total, desc=f"[{i + 1}/{total}] {q['question_type']}")

            # 1. clear DB + enroll this question's haystack
            try:
                prep = requests.post(
                    f"{FASTAPI_URL}/admin/longmemeval/prepare",
                    json={
                        "split": split,
                        "question_id": q["question_id"],
                        "enable_kg": bool(enable_kg),
                    },
                    timeout=3600,
                )
            except Exception as e:
                log.append(f"[{i + 1}/{total}] PREP exception: {e}")
                continue
            if prep.status_code != 200:
                log.append(f"[{i + 1}/{total}] PREP {prep.status_code}: {prep.text[:200]}")
                continue

            # 2. build context for the question
            try:
                ret = requests.post(
                    f"{FASTAPI_URL}/rag/retrieve",
                    data={"question": q["question"]},
                    timeout=180,
                )
            except Exception as e:
                log.append(f"[{i + 1}/{total}] RETRIEVE exception: {e}")
                continue
            if ret.status_code != 200:
                log.append(f"[{i + 1}/{total}] RETRIEVE {ret.status_code}: {ret.text[:200]}")
                continue

            prompt = ret.json().get("recommended_prompt", "")
            if not prompt:
                log.append(f"[{i + 1}/{total}] empty recommended_prompt")
                continue

            # 3. run Qwen on the prompt
            try:
                hypothesis = _qwen_answer(prompt, max_new_tokens=300)
            except Exception as e:
                log.append(f"[{i + 1}/{total}] QWEN exception: {e}")
                continue

            # 4. append result
            out_f.write(json.dumps({
                "question_id":   q["question_id"],
                "hypothesis":    hypothesis,
                "question":      q["question"],
                "answer":        q["answer"],
                "question_type": q["question_type"],
                "question_date": q.get("question_date", ""),
            }) + "\n")
            out_f.flush()

            log.append(
                f"[{i + 1}/{total}] {q['question_type']} | "
                f"hyp: {hypothesis[:80].replace(chr(10), ' ')}"
            )
    finally:
        out_f.close()

    log.append("")
    log.append(f"Done. Results saved to {out_path}")
    return "\n".join(log)


def run_longmemeval_llm_only(split, n_questions, progress=gr.Progress()):
    """Run the benchmark with Qwen alone: full haystack stuffed into the prompt, no RAG.

    Always routes the answer call to the BACKEND's loaded model
    (Qwen3-14B-AWQ via vLLM) so this baseline shares the exact same answerer
    as the RAG path — keeping the comparison apples-to-apples. Requires the
    backend extractor mode to be 'local'.
    """
    try:
        questions = _lme_questions(split, n_questions)
    except Exception as e:
        return f"Failed to load split: {e}"

    total = len(questions)
    if total == 0:
        return f"No questions found in split '{split}'."

    # Preflight: refuse to start when the backend has no LLM loaded — otherwise
    # every /llm/generate call would 409 and we'd write zero results.
    if _backend_extractor_mode() != "local":
        return (
            "LLM-only baseline aborted: backend extractor mode is not 'local'.\n"
            "Activate Local Qwen-14B on the backend first (Step 4 above), "
            "then re-run this baseline."
        )

    out_path, out_f = _lme_open_output("llm_only", split)
    log = [
        f"Running LongMemEval (LLM-only baseline) on '{split}' — {total} questions",
        f"  Answerer: backend Qwen3-14B-AWQ via vLLM (max_model_len=32k).",
        f"  Haystack truncated to ~80k chars to leave room for prompt + answer.",
        f"  Output: {out_path}",
        "",
    ]

    try:
        for i, q in enumerate(questions):
            progress(i / total, desc=f"[{i + 1}/{total}] {q['question_type']}")

            try:
                hs = requests.get(
                    f"{FASTAPI_URL}/admin/longmemeval/haystack",
                    params={"split": split, "question_id": q["question_id"]},
                    timeout=180,
                )
            except Exception as e:
                log.append(f"[{i + 1}/{total}] HAYSTACK exception: {e}")
                continue
            if hs.status_code != 200:
                log.append(f"[{i + 1}/{total}] HAYSTACK {hs.status_code}: {hs.text[:200]}")
                continue

            data = hs.json()
            chat_history = _flatten_haystack(
                data.get("haystack_sessions", []),
                data.get("haystack_dates", []),
            )
            prompt = _LME_LLM_ONLY_PROMPT.format(
                chat_history=chat_history,
                question_date=q.get("question_date", "unknown"),
                question=q["question"],
            )

            try:
                # Force-route through the backend so this baseline always uses
                # Qwen3-14B-AWQ via vLLM, matching the RAG path's answerer.
                hypothesis = _answer_with_backend_local(prompt, max_new_tokens=300)
            except Exception as e:
                log.append(f"[{i + 1}/{total}] QWEN exception: {e}")
                continue

            out_f.write(json.dumps({
                "question_id":   q["question_id"],
                "hypothesis":    hypothesis,
                "question":      q["question"],
                "answer":        q["answer"],
                "question_type": q["question_type"],
                "question_date": q.get("question_date", ""),
            }) + "\n")
            out_f.flush()

            log.append(
                f"[{i + 1}/{total}] {q['question_type']} | "
                f"hyp: {hypothesis[:80].replace(chr(10), ' ')}"
            )
    finally:
        out_f.close()

    log.append("")
    log.append(f"Done. Results saved to {out_path}")
    return "\n".join(log)


# ==========================================
# Step 6: MLC-SLM Task 2 Benchmark
# ==========================================
# Multi-speaker, multi-lingual audio conversations + multi-choice questions.
# Per conversation: clear DB → enroll the .wav (diarization + STT + KG) →
# answer each A/B/C/D question via /llm/generate.
#
# LLM-only baseline path: skip enrollment, read the reference .txt alongside
# the .wav, stuff it into the prompt as the conversation transcript.

import re as _re_mlcslm

_MLCSLM_LETTER_RE = _re_mlcslm.compile(r"\b([ABCD])\b")

_MLCSLM_RAG_PROMPT = """You will answer a multiple-choice question about a recorded conversation.

CONTEXT (retrieved from the conversation memory):
{context}

QUESTION: {question_stem}

OPTIONS:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Output ONLY the single letter (A, B, C, or D) of the best option. \
Do NOT explain. Do NOT repeat the option text.
ANSWER:"""

_MLCSLM_LLM_ONLY_PROMPT = """You will answer a multiple-choice question about a recorded conversation.

CONVERSATION TRANSCRIPT:
{transcript}

QUESTION: {question_stem}

OPTIONS:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Output ONLY the single letter (A, B, C, or D) of the best option. \
Do NOT explain. Do NOT repeat the option text.
ANSWER:"""


def _mlcslm_open_output(mode_tag: str):
    """Create the MLC-SLM results dir + open a new JSONL for incremental writes."""
    os.makedirs(MLCSLM_RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(MLCSLM_RESULTS_DIR) / f"mlcslm_{mode_tag}_{stamp}.jsonl"
    return path, open(path, "w")


def _mlcslm_fetch_qa(qa_jsonl: str, limit: int) -> List[dict]:
    """Fetch QA records from the backend (path is backend-side)."""
    r = requests.get(
        f"{FASTAPI_URL}/admin/mlcslm/qa",
        params={"qa_jsonl": qa_jsonl, "limit": int(limit)},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/admin/mlcslm/qa {r.status_code}: {r.text[:300]}")
    return r.json().get("records", []) or []


def _mlcslm_prepare(audio_path: str, audio_root: str) -> dict:
    r = requests.post(
        f"{FASTAPI_URL}/admin/mlcslm/prepare",
        json={"audio_path": audio_path, "audio_root": audio_root, "clear_first": True},
        timeout=3600,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/admin/mlcslm/prepare {r.status_code}: {r.text[:300]}")
    return r.json()


def _mlcslm_fetch_transcript(audio_path: str, audio_root: str, max_chars: int) -> str:
    r = requests.get(
        f"{FASTAPI_URL}/admin/mlcslm/transcript",
        params={"audio_path": audio_path, "audio_root": audio_root, "max_chars": int(max_chars)},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/admin/mlcslm/transcript {r.status_code}: {r.text[:300]}")
    return r.json().get("transcript", "") or ""


def _mlcslm_option_text(question: dict, label: str) -> str:
    for opt in question.get("options", []):
        if str(opt.get("label", "")).strip().upper() == label:
            return str(opt.get("text", "")).strip()
    return ""


def _mlcslm_parse_letter(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = _MLCSLM_LETTER_RE.search(raw.strip())
    return m.group(1) if m else None


def _mlcslm_qwen_letter(prompt: str) -> str:
    """Route through the backend's Qwen3-14B (same answerer as the RAG path)."""
    return _answer_with_backend_local(prompt, max_new_tokens=8)


def run_mlcslm_rag(qa_jsonl_str, audio_root_str, n_audios, progress=gr.Progress()):
    """Run MLC-SLM Task 2 with full SpeakerRAG. Paths are backend-side."""
    if _backend_extractor_mode() != "local":
        return (
            "MLC-SLM aborted: backend extractor mode is not 'local'.\n"
            "Activate Local Qwen-14B (Step 4) first."
        )

    qa_jsonl_str = (qa_jsonl_str or "").strip()
    audio_root_str = (audio_root_str or "").strip()
    if not qa_jsonl_str:
        return "QA JSONL path is empty."
    if not audio_root_str:
        return "Audio root path is empty."

    try:
        n_audios = int(n_audios)
    except (TypeError, ValueError):
        n_audios = -1

    try:
        records = _mlcslm_fetch_qa(qa_jsonl_str, n_audios)
    except Exception as e:
        return f"Failed to fetch QA from backend: {e}"

    total_audios = len(records)
    total_q = sum(len(r.get("questions", [])) for r in records)
    out_path, out_f = _mlcslm_open_output("rag")
    log = [
        f"Running MLC-SLM Task 2 (SpeakerRAG) — {total_audios} audios, {total_q} questions",
        f"  Answerer: backend Qwen3-14B-AWQ via vLLM.",
        f"  Backend QA JSONL: {qa_jsonl_str}",
        f"  Backend audio root: {audio_root_str}",
        f"  Output: {out_path}",
        "",
    ]

    n_ok = 0
    n_fail = 0
    n_correct = 0
    try:
        for i, rec in enumerate(records):
            progress(i / max(total_audios, 1), desc=f"[{i + 1}/{total_audios}]")
            jsonl_path = rec.get("path", "")
            questions = rec.get("questions", []) or []
            if not questions:
                continue

            # Clear DB + enroll the conversation, both delegated to the backend.
            try:
                prep = _mlcslm_prepare(jsonl_path, audio_root_str)
                wav_name = Path(prep.get("wav_path", "") or jsonl_path).name
            except Exception as e:
                n_fail += len(questions)
                log.append(f"[{i + 1}/{total_audios}] PREPARE {jsonl_path} FAIL: {e}")
                continue

            log.append(f"[{i + 1}/{total_audios}] enrolled {wav_name}; "
                       f"{len(questions)} questions")

            for q in questions:
                try:
                    ret = requests.post(
                        f"{FASTAPI_URL}/rag/retrieve",
                        data={"question": q.get("question_stem", "")},
                        timeout=300,
                    )
                    if ret.status_code != 200:
                        raise RuntimeError(f"/rag/retrieve {ret.status_code}")
                    fact_sheet = ret.json().get("fact_sheet", "") or ""
                    prompt = _MLCSLM_RAG_PROMPT.format(
                        context=fact_sheet,
                        question_stem=q.get("question_stem", ""),
                        option_a=_mlcslm_option_text(q, "A"),
                        option_b=_mlcslm_option_text(q, "B"),
                        option_c=_mlcslm_option_text(q, "C"),
                        option_d=_mlcslm_option_text(q, "D"),
                    )
                    raw = _mlcslm_qwen_letter(prompt)
                    predicted = _mlcslm_parse_letter(raw)
                except Exception as e:
                    n_fail += 1
                    log.append(f"    {q.get('question_id', '?')} FAIL: {e}")
                    continue

                gold = str(q.get("correct_answer", "")).strip().upper() or None
                is_correct = (predicted is not None and predicted == gold)

                out_f.write(json.dumps({
                    "audio_path":       jsonl_path,
                    "question_id":      q.get("question_id", ""),
                    "question_stem":    q.get("question_stem", ""),
                    "options":          q.get("options", []),
                    "correct_answer":   gold,
                    "predicted_answer": predicted,
                    "raw_response":     raw,
                    "is_correct":       is_correct,
                }) + "\n")
                out_f.flush()
                n_ok += 1
                if is_correct:
                    n_correct += 1

            running_acc = n_correct / max(n_ok, 1)
            log.append(
                f"[{i + 1}/{total_audios}] {wav_path.name}: "
                f"{len(questions)} qs; running acc={running_acc:.3f}"
            )
    finally:
        out_f.close()

    final_acc = n_correct / max(n_ok, 1)
    log.append("")
    log.append(f"Done. {n_ok} answered, {n_correct} correct, {n_fail} failed. "
               f"Accuracy={final_acc:.4f}")
    log.append(f"Results: {out_path}")
    return "\n".join(log)


def run_mlcslm_llm_only(qa_jsonl_str, audio_root_str, n_audios, max_transcript_chars,
                        progress=gr.Progress()):
    """Run MLC-SLM Task 2 LLM-only. Paths are backend-side."""
    if _backend_extractor_mode() != "local":
        return (
            "MLC-SLM LLM-only aborted: backend extractor mode is not 'local'.\n"
            "Activate Local Qwen-14B (Step 4) first."
        )

    qa_jsonl_str = (qa_jsonl_str or "").strip()
    audio_root_str = (audio_root_str or "").strip()
    if not qa_jsonl_str:
        return "QA JSONL path is empty."
    if not audio_root_str:
        return "Audio root path is empty."

    try:
        max_chars = int(max_transcript_chars)
    except (TypeError, ValueError):
        max_chars = 80000

    try:
        n_audios = int(n_audios)
    except (TypeError, ValueError):
        n_audios = -1

    try:
        records = _mlcslm_fetch_qa(qa_jsonl_str, n_audios)
    except Exception as e:
        return f"Failed to fetch QA from backend: {e}"

    total_audios = len(records)
    total_q = sum(len(r.get("questions", [])) for r in records)
    out_path, out_f = _mlcslm_open_output("llm_only")
    log = [
        f"Running MLC-SLM Task 2 (LLM-only) — {total_audios} audios, {total_q} questions",
        f"  Answerer: backend Qwen3-14B-AWQ via vLLM.",
        f"  Backend QA JSONL: {qa_jsonl_str}",
        f"  Backend audio root: {audio_root_str}",
        f"  Transcript truncation: {max_chars} chars.",
        f"  Output: {out_path}",
        "",
    ]

    n_ok = 0
    n_fail = 0
    n_correct = 0
    try:
        for i, rec in enumerate(records):
            progress(i / max(total_audios, 1), desc=f"[{i + 1}/{total_audios}]")
            jsonl_path = rec.get("path", "")
            questions = rec.get("questions", []) or []
            if not questions:
                continue

            try:
                transcript = _mlcslm_fetch_transcript(jsonl_path, audio_root_str, max_chars)
            except Exception as e:
                n_fail += len(questions)
                log.append(f"[{i + 1}/{total_audios}] TRANSCRIPT FAIL ({jsonl_path}): {e}")
                continue

            for q in questions:
                try:
                    prompt = _MLCSLM_LLM_ONLY_PROMPT.format(
                        transcript=transcript,
                        question_stem=q.get("question_stem", ""),
                        option_a=_mlcslm_option_text(q, "A"),
                        option_b=_mlcslm_option_text(q, "B"),
                        option_c=_mlcslm_option_text(q, "C"),
                        option_d=_mlcslm_option_text(q, "D"),
                    )
                    raw = _mlcslm_qwen_letter(prompt)
                    predicted = _mlcslm_parse_letter(raw)
                except Exception as e:
                    n_fail += 1
                    log.append(f"    {q.get('question_id', '?')} FAIL: {e}")
                    continue

                gold = str(q.get("correct_answer", "")).strip().upper() or None
                is_correct = (predicted is not None and predicted == gold)

                out_f.write(json.dumps({
                    "audio_path":       jsonl_path,
                    "question_id":      q.get("question_id", ""),
                    "question_stem":    q.get("question_stem", ""),
                    "options":          q.get("options", []),
                    "correct_answer":   gold,
                    "predicted_answer": predicted,
                    "raw_response":     raw,
                    "is_correct":       is_correct,
                }) + "\n")
                out_f.flush()
                n_ok += 1
                if is_correct:
                    n_correct += 1

            running_acc = n_correct / max(n_ok, 1)
            log.append(
                f"[{i + 1}/{total_audios}] {wav_path.name}: "
                f"{len(questions)} qs; running acc={running_acc:.3f}"
            )
    finally:
        out_f.close()

    final_acc = n_correct / max(n_ok, 1)
    log.append("")
    log.append(f"Done. {n_ok} answered, {n_correct} correct, {n_fail} failed. "
               f"Accuracy={final_acc:.4f}")
    log.append(f"Results: {out_path}")
    return "\n".join(log)


# ==========================================
# Gradio UI Layout
# ==========================================
with gr.Blocks(title="Speaker RAG Engine Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ Speaker RAG Engine Demo")
    gr.Markdown(
        "This UI demonstrates a decoupled architecture. The backend extracts audio context "
        "and builds a Fact Sheet, which is then passed to whichever Text LLM the user picked "
        "in **LLM Settings** below: either the UI-side `Qwen2-7B-Instruct` (Claude API mode) "
        "or the backend's `Qwen3-14B` (Local mode)."
    )
    
    with gr.Accordion("🔑 LLM Settings (KG extraction backend)", open=True):
        gr.Markdown(
            "Choose how the knowledge graph extracts entities and relations during "
            "enrollment. Two backends are supported:\n\n"
            "- **Claude API (Haiku + Sonnet)** — fastest, highest-quality, but billed per "
            "token. Requires an Anthropic API key. The UI's Qwen2-7B-Instruct handles "
            "fact-sheet answering (lazy-loaded on first question).\n"
            "- **Local Qwen3-14B (8-bit)** — free, runs entirely on the backend GPU "
            "(~16 GB VRAM). The same model handles fact-sheet answering, so the UI's "
            "Qwen2-7B is unloaded on switch. First activation downloads ~28 GB from "
            "HuggingFace; subsequent restarts re-load from cache in ~30 s.\n\n"
            "The Anthropic key (when used) is kept in the API process memory only — "
            "it is **not persisted to disk** and is **never returned** by any endpoint."
        )
        with gr.Row():
            llm_key_input = gr.Textbox(
                label="Anthropic API key",
                placeholder="sk-ant-...",
                type="password",
                scale=4,
            )
            llm_set_btn = gr.Button("Activate Claude API", variant="primary", scale=1)
        with gr.Row():
            llm_local_btn = gr.Button("Activate Local Qwen3-14B", variant="primary", scale=2)
            llm_clear_btn = gr.Button("Disable extractor", variant="stop", scale=1)
            llm_refresh_btn = gr.Button("🔄 Refresh", scale=1)
        llm_status = gr.Textbox(
            label="Extractor status",
            interactive=False,
            lines=2,
            value=get_mode_status_ui(),
        )

        llm_set_btn.click(
            fn=activate_claude_mode,
            inputs=[llm_key_input],
            outputs=[llm_status, llm_key_input],
        )
        llm_local_btn.click(
            fn=activate_local_mode,
            inputs=[],
            outputs=[llm_status],
        )
        llm_clear_btn.click(
            fn=deactivate_extractor,
            inputs=[],
            outputs=[llm_status],
        )
        llm_refresh_btn.click(
            fn=get_mode_status_ui,
            inputs=[],
            outputs=[llm_status],
        )

    with gr.Accordion("⚙️ Step 1: Database Management (Enrollment)", open=False):
        gr.Markdown("Enroll known speakers into the ChromaDB vector database before querying.")
        with gr.Row():
            with gr.Column():
                enroll_audio = gr.File(label="Upload Audio to Enroll", file_count="multiple", type="filepath")
                enroll_ids = gr.Textbox(
                    label="User IDs (comma-separated)", 
                    placeholder="e.g., Speaker_A, Speaker_B (Leave blank for random unique UUIDs)"
                )
                with gr.Row():
                    enroll_btn = gr.Button("Enroll to RAG")
                    list_btn = gr.Button("Show Enrolled Users", variant="secondary")
                    clear_btn = gr.Button("Clear Entire Database", variant="stop")
                    meld_btn = gr.Button("🚀 Ingest MELD (Admin)", variant="primary")
                with gr.Row():
                    vox1_per_speaker = gr.Number(
                        label="VoxCeleb1: utterances per speaker",
                        value=10, precision=0, scale=1,
                    )
                    vox1_btn = gr.Button("🎤 Ingest VoxCeleb1 (Admin)", variant="primary", scale=2)

            with gr.Column():
                enroll_output = gr.Textbox(label="Enrollment / Database Status", lines=10)

        enroll_btn.click(fn=enroll_rag, inputs=[enroll_audio, enroll_ids], outputs=enroll_output)
        list_btn.click(fn=list_enrolled_users, inputs=[], outputs=enroll_output)
        clear_btn.click(fn=clear_rag, inputs=[], outputs=enroll_output)
        meld_btn.click(fn=ingest_meld_ui, inputs=[], outputs=enroll_output)
        vox1_btn.click(fn=ingest_voxceleb_ui, inputs=[vox1_per_speaker], outputs=enroll_output)

        with gr.Accordion("🕸️ Knowledge Graph Visualization", open=False):
            gr.Markdown(
                "Interactive graph of the speaker-entity relationships built during enrollment. "
                "**Speaker nodes** (blue boxes) connect to **entity nodes** (coloured dots) via dashed "
                "MENTIONS edges. Thick solid edges between speakers = KNOWS (shared dialogue). "
                "Utterance nodes are hidden — too many to display. "
                "**Tip:** scroll to zoom · drag nodes · hover for details · use nav buttons bottom-right."
            )
            with gr.Row():
                kg_max_entities = gr.Slider(
                    label="Max entity nodes shown",
                    minimum=5, maximum=100, value=40, step=5, scale=3,
                )
                kg_refresh_btn = gr.Button("🔄 Refresh Graph", variant="secondary", scale=1)
            kg_plot  = gr.HTML()
            kg_stats = gr.Textbox(label="Graph Statistics", interactive=False, lines=1)

            kg_refresh_btn.click(
                fn=get_graph_viz,
                inputs=[kg_max_entities],
                outputs=[kg_plot, kg_stats],
            )
        
    with gr.Accordion("🎙️ Step 1b: Conversation Enrollment (Diarization)", open=False):
        gr.Markdown(
            "Upload a full multi-speaker conversation (e.g. a Teams call, phone recording). "
            "The backend diarizes the audio, clusters speakers, matches them against enrolled "
            "identities, and enrolls each segment.\n\n"
            "- **VTT file** (recommended for Teams/Meet): upload the `.vtt` transcript file. "
            "Speaker names, timestamps, and transcriptions are extracted directly — pyannote "
            "diarization is skipped entirely.\n"
            "- **Speaker Map** (optional JSON): manual mapping of pyannote labels to names. "
            "Ignored if a VTT file is provided.\n"
            "- **Num speakers** (optional): hint for the clustering algorithm. "
            "Ignored if a VTT file is provided."
        )
        with gr.Row():
            with gr.Column():
                conv_audio = gr.File(label="Upload Conversation Audio", file_count="single", type="filepath")
                conv_vtt = gr.File(label="Upload VTT Transcript (optional, e.g. from Teams)", file_count="single", type="filepath")
                conv_speaker_map = gr.Textbox(
                    label="Speaker Map (optional JSON, ignored if VTT provided)",
                    placeholder='e.g. {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}',
                )
                conv_num_speakers = gr.Number(
                    label="Number of Speakers (optional, ignored if VTT provided)",
                    precision=0,
                    value=None,
                )
                conv_btn = gr.Button("🎙️ Diarize & Enroll Conversation", variant="primary")
            with gr.Column():
                conv_output = gr.Textbox(label="Conversation Enrollment Result", lines=12)

        def enroll_conversation(audio_path, vtt_path, speaker_map_str, num_spk):
            if not audio_path:
                return "Please upload a conversation audio file."

            files = [('audio', (os.path.basename(audio_path), open(audio_path, 'rb'), 'audio/wav'))]
            if vtt_path:
                files.append(('vtt_file', (os.path.basename(vtt_path), open(vtt_path, 'rb'), 'text/vtt')))

            data = {}
            if not vtt_path:
                if speaker_map_str and speaker_map_str.strip():
                    data['speaker_map'] = speaker_map_str.strip()
                if num_spk is not None and int(num_spk) >= 2:
                    data['num_speakers'] = int(num_spk)

            try:
                response = requests.post(f"{FASTAPI_URL}/rag/index_conversation", files=files, data=data)
                if response.status_code == 200:
                    res = response.json()
                    lines = [
                        f"Dialogue ID: {res.get('dialogue_id', 'N/A')}",
                        f"Enrolled Segments: {res.get('enrolled_segments', 0)}",
                        f"Identity Map:",
                    ]
                    for cluster, identity in res.get('identity_map', {}).items():
                        lines.append(f"  {cluster} -> {identity}")
                    return "\n".join(lines)
                else:
                    return f"Error {response.status_code}:\n{response.text}"
            except Exception as e:
                return f"An error occurred: {str(e)}"

        conv_btn.click(
            fn=enroll_conversation,
            inputs=[conv_audio, conv_vtt, conv_speaker_map, conv_num_speakers],
            outputs=conv_output,
        )

    with gr.Accordion("🔍 Step 2: Query / Probe Audio", open=True):
        gr.Markdown(
            "Ask a question with or without audio:\n"
            "- **With audio**: the backend identifies the speaker(s) and filters context by identity.\n"
            "- **Text only**: the backend scans the question for enrolled speaker names; if any match, it filters by them, otherwise it runs an unfiltered semantic search.\n\n"
            "Prior Q/A turns are stored client-side and sent along with each new question so the LLM can resolve follow-up references. "
            "Use **💾 Save to Memory** to enroll the current chat into the long-term store (semantic + KG, like an audio dialogue) — "
            "future queries will then be able to retrieve from it. Use **Reset Memory** to clear the client-side history only."
        )

        conversation_state = gr.State([])

        with gr.Row():
            with gr.Column():
                rag_audio = gr.File(label="Upload Probe Audio Files (optional)", file_count="multiple", type="filepath")
                rag_question = gr.Textbox(
                    label="Question",
                    placeholder="e.g., Are the speakers the same? What did Alice say about the project?"
                )
                with gr.Row():
                    rag_btn = gr.Button("Extract Context & Ask LLM", variant="primary")
                    save_chat_btn = gr.Button("💾 Save to Memory", variant="secondary")
                    reset_btn = gr.Button("🧹 Reset Memory", variant="stop")

            with gr.Column():
                rag_output = gr.Textbox(label="LLM Response", lines=6)
                history_display = gr.Textbox(label="Conversation Memory", value="(no prior turns)", lines=8, interactive=False)
                save_chat_status = gr.Textbox(label="Save to Memory Status", value="", lines=2, interactive=False)
                with gr.Accordion("⚙️ View Backend Context (Debug)", open=False):
                    rag_context = gr.Textbox(label="RAG Fact Sheet (Returned by API)", lines=15)

        rag_btn.click(
            fn=ask_rag,
            inputs=[rag_audio, rag_question, conversation_state],
            outputs=[rag_output, rag_context, conversation_state, history_display],
        )
        save_chat_btn.click(
            fn=save_chat_to_memory,
            inputs=[conversation_state],
            outputs=save_chat_status,
        )
        reset_btn.click(
            fn=reset_history,
            inputs=[],
            outputs=[conversation_state, history_display],
        )

    with gr.Accordion("📊 Step 3: Speaker Verification Evaluation (MELD dev)", open=False):
        gr.Markdown(
            "Run the three EER-based speaker-verification experiments on MELD dev pairs.\n\n"
            "**Recommended workflow:**\n"
            "1. Click **Clear Entire Database** in Step 1.\n"
            "2. Run **Experiment 1** (Qwen2-Audio direct — does not touch the DB).\n"
            "3. Run **Experiment 2** (RAG without enrollment — DB must be empty).\n"
            "4. Click **🚀 Ingest MELD (Admin)** in Step 1 to enroll the MELD train split.\n"
            "5. Run **Experiment 3** (RAG with MELD train enrolled).\n\n"
            "Pairs are generated once (1000 same-speaker + 1000 different-speaker, seed=42) "
            "and cached. Tick *Regenerate pairs* to rebuild. "
            "Use *Number of pairs* to limit the run for quick checks; -1 means all 2000."
        )
        with gr.Row():
            eval_n_pairs = gr.Number(label="Number of pairs (-1 = all 2000)", value=200, precision=0)
            eval_regen = gr.Checkbox(label="Regenerate pairs", value=False)

        with gr.Row():
            exp1_btn = gr.Button("Experiment 1: Qwen2-Audio direct", variant="primary")
            exp2_btn = gr.Button("Experiment 2: RAG (empty DB) + text LLM", variant="primary")
            exp3_btn = gr.Button("Experiment 3: RAG (MELD enrolled) + text LLM", variant="primary")

        eval_output = gr.Textbox(label="Evaluation Result", lines=10)

        exp1_btn.click(fn=run_exp1, inputs=[eval_n_pairs, eval_regen], outputs=eval_output)
        exp2_btn.click(fn=run_exp2, inputs=[eval_n_pairs, eval_regen], outputs=eval_output)
        exp3_btn.click(fn=run_exp3, inputs=[eval_n_pairs, eval_regen], outputs=eval_output)

    with gr.Accordion("📊 Step 4: Speaker Verification Evaluation (VoxCeleb1 Vox1-O)", open=False):
        gr.Markdown(
            "Run the same three EER experiments on the VoxCeleb1 Vox1-O trial list — "
            "much cleaner audio than MELD and the canonical ASV benchmark used in the "
            "Speech-Aware LLM paper.\n\n"
            "**Recommended workflow:**\n"
            "1. Click **Clear Entire Database** in Step 1.\n"
            "2. Run **Vox1 Experiment 1** (Qwen2-Audio direct — does not touch the DB).\n"
            "3. Run **Vox1 Experiment 2** (RAG without enrollment — DB must be empty).\n"
            "4. Click **🎤 Ingest VoxCeleb1 (Admin)** in Step 1 to enroll the 40 test "
            "speakers (probe files in Vox1-O are excluded automatically).\n"
            "5. Run **Vox1 Experiment 3** (RAG with VoxCeleb1 enrolled).\n\n"
            "Vox1-O has 37,720 pairs total. Default subset is 200 (balanced same/diff). "
            "Set *Number of pairs* (-1 = all 37,720) and click **Build / Rebuild pairs file** "
            "first — this must be done before Ingesting VoxCeleb1 so probe files are correctly excluded."
        )
        with gr.Row():
            vox1_eval_n_pairs = gr.Number(
                label="Number of pairs (-1 = all 37,720)",
                value=200, precision=0,
            )
            vox1_build_btn = gr.Button("Build / Rebuild pairs file", variant="secondary")

        with gr.Row():
            vox1_exp1_btn = gr.Button("Vox1 Experiment 1: Qwen2-Audio direct", variant="primary")
            vox1_exp2_btn = gr.Button("Vox1 Experiment 2: RAG (empty DB) + text LLM", variant="primary")
            vox1_exp3_btn = gr.Button("Vox1 Experiment 3: RAG (VoxCeleb1 enrolled) + text LLM", variant="primary")

        vox1_eval_output = gr.Textbox(label="VoxCeleb1 Evaluation Result", lines=10)

        vox1_build_btn.click(fn=build_vox1_pairs_ui,
                             inputs=[vox1_eval_n_pairs],
                             outputs=vox1_eval_output)
        vox1_exp1_btn.click(fn=run_vox1_exp1,
                            inputs=[vox1_eval_n_pairs, gr.State(False)],
                            outputs=vox1_eval_output)
        vox1_exp2_btn.click(fn=run_vox1_exp2,
                            inputs=[vox1_eval_n_pairs, gr.State(False)],
                            outputs=vox1_eval_output)
        vox1_exp3_btn.click(fn=run_vox1_exp3,
                            inputs=[vox1_eval_n_pairs, gr.State(False)],
                            outputs=vox1_eval_output)

    with gr.Accordion("📊 Step 5: LongMemEval Benchmark", open=False):
        gr.Markdown(
            "Evaluate the long-term-memory pipeline on the **LongMemEval** benchmark "
            "(Wu et al., ICLR 2025). For each instance, the runner clears the database, "
            "enrolls the question's `haystack_sessions`, retrieves context, and asks Qwen the question. "
            "Results are written as a JSONL file compatible with the official `evaluate_qa.py`.\n\n"
            "**Modes:**\n"
            "- **Run with SpeakerRAG** — full pipeline: clear DB → enroll haystack → RAG-built context → Qwen.\n"
            "- **Run LLM-only baseline** — Qwen with the full haystack stuffed into the prompt, no RAG. "
            "Note: Qwen2-7B-Instruct has ~32k token context, so the `s_cleaned` and `m_cleaned` "
            "splits will be truncated; `oracle` typically fits.\n\n"
            "**KG extraction toggle** — when ON, every enrolled session goes through Haiku entity / "
            "Sonnet relation extraction (slow but exercises the full system). Turn OFF for a fast "
            "semantic-retrieval-only run.\n\n"
            "Results are saved under `tests/longmemeval_results/`. Copy the JSONL out of the container "
            "and run, in your local conda env:\n"
            "```\n"
            "python3 evaluate_qa.py gpt-4o longmemeval_<mode>_<split>_<timestamp>.jsonl "
            "../../data/longmemeval_oracle.json\n"
            "```"
        )

        with gr.Row():
            lme_split = gr.Dropdown(
                label="Split",
                choices=["oracle", "s_cleaned", "m_cleaned"],
                value="oracle",
                scale=2,
            )
            lme_n_questions = gr.Number(
                label="Number of questions (-1 = all)",
                value=20, precision=0, scale=1,
            )
            lme_enable_kg = gr.Checkbox(
                label="Enable KG extraction during enrollment",
                value=True, scale=1,
            )

        with gr.Row():
            lme_rag_btn = gr.Button("Run with SpeakerRAG", variant="primary")
            lme_baseline_btn = gr.Button("Run LLM-only baseline", variant="secondary")

        lme_output = gr.Textbox(label="Benchmark Progress / Log", lines=20)

        lme_rag_btn.click(
            fn=run_longmemeval_rag,
            inputs=[lme_split, lme_n_questions, lme_enable_kg],
            outputs=lme_output,
        )
        lme_baseline_btn.click(
            fn=run_longmemeval_llm_only,
            inputs=[lme_split, lme_n_questions],
            outputs=lme_output,
        )

    with gr.Accordion("🎙️ Step 6: MLC-SLM Task 2 Benchmark", open=False):
        gr.Markdown(
            "Evaluate the pipeline on the **MLC-SLM Challenge Task 2 — Multilingual "
            "Conversational Speech Understanding**. For each multi-speaker phone-call "
            "audio:\n"
            "- **Run with SpeakerRAG** — clear DB → upload .wav to "
            "`/rag/index_conversation` (pyannote diarization + STT + KG extraction) → "
            "for each question, hit `/rag/retrieve` and ask Qwen3-14B to pick A/B/C/D.\n"
            "- **Run LLM-only baseline** — skip enrollment; read the reference `.txt` "
            "transcript that ships alongside each `.wav` and stuff it into the prompt. "
            "Same answerer (Qwen3-14B-AWQ via vLLM), so the only diff between modes is "
            "whether retrieval is involved.\n\n"
            "**Path resolution:** the QA JSONL uses paths like "
            "`mlc-slm-2nd-dev/English_Australian/1127_015_phone.wav` that don't match "
            "the on-disk layout. We resolve by basename: every `.wav` under the audio "
            "root is indexed once at the start of each run.\n\n"
            "**Prerequisite:** backend extractor mode must be `local` (Step 4)."
        )

        with gr.Row():
            mlcslm_qa_jsonl = gr.Textbox(
                label="QA JSONL path",
                value="/app/tests/mlcslm_data/mlcslm_2nd_dev_qa_300_subset.jsonl",
                scale=3,
            )
            mlcslm_audio_root = gr.Textbox(
                label="Audio root (folder containing language subdirs)",
                value="/corpora/mlc-slm-2nd-workshop-data/dev_data",
                scale=3,
            )
        with gr.Row():
            mlcslm_n_audios = gr.Number(
                label="Number of audios (-1 = all)",
                value=5, precision=0, scale=1,
            )
            mlcslm_max_transcript_chars = gr.Number(
                label="LLM-only: max transcript chars",
                value=80000, precision=0, scale=1,
            )

        with gr.Row():
            mlcslm_rag_btn = gr.Button("Run with SpeakerRAG", variant="primary")
            mlcslm_baseline_btn = gr.Button("Run LLM-only baseline", variant="secondary")

        mlcslm_output = gr.Textbox(label="MLC-SLM Progress / Log", lines=25)

        mlcslm_rag_btn.click(
            fn=run_mlcslm_rag,
            inputs=[mlcslm_qa_jsonl, mlcslm_audio_root, mlcslm_n_audios],
            outputs=mlcslm_output,
        )
        mlcslm_baseline_btn.click(
            fn=run_mlcslm_llm_only,
            inputs=[mlcslm_qa_jsonl, mlcslm_audio_root, mlcslm_n_audios, mlcslm_max_transcript_chars],
            outputs=mlcslm_output,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)