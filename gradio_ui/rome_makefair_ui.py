"""
SpeakerRAGent — Rome Maker Faire booth UI.

A stripped-down, visitor-facing front-end for the demo booth. It is a *thin*
client: every model call goes to the FastAPI backend over the existing
SSH-forwarded connection (`FASTAPI_URL`), exactly like `app_ui.py`. Nothing is
loaded locally, so this file imports neither torch nor transformers and runs
happily on a laptop with no GPU.

Booth flow
----------
1. **Registrati** — the visitor records ~15 s into the mic: their name, where
   they are from, and one small fact or story. Optionally the operator first
   presses *Ascolta* so the machine transcribes the clip and reports whether it
   already knows that voice ("Non conosco ancora questa voce"). Then a name is
   typed and the same clip is enrolled through `POST /rag/index`.
2. **Chiedi** — anyone records a *new* clip, which is sent as a probe alongside
   a question to `POST /rag/retrieve`. The backend identifies the speaker from
   voice alone and assembles a fact sheet. The same answer LLM is then asked
   the question twice — once with the fact sheet and once with nothing — so the
   two panels differ only in whether memory was available.
3. **La memoria** — the knowledge graph, refreshable and optionally
   auto-refreshing, for the second screen.
4. **Regia** — operator controls: backend status, roster, bulk seeding of
   colleague recordings, and the (destructive) database reset.

Endpoint contract — this module only *consumes* the existing API:
    GET    /health
    GET    /admin/extractor_mode
    GET    /rag/enrolled
    GET    /rag/graph_viz
    POST   /rag/index
    POST   /rag/retrieve
    POST   /llm/generate
    DELETE /rag/clear

Browser microphone capture requires a secure context, so open this UI on
http://localhost:7861 from the same machine that runs it (a LAN IP will make
the browser refuse mic permission).
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests
from pyvis.network import Network

# --- Configuration ---
# Unchanged from app_ui.py: the backend is reached through the SSH port forward.
FASTAPI_URL = "http://127.0.0.1:8006"  # fbk server (requires SSH Port Forwarding)

# Enrollment runs Voxtral + Titanet + MiniLM and then flushes the KG extraction
# queue through Qwen3-14B, so it is the slowest call in the booth by far.
ENROLL_TIMEOUT = 900
# A conversation runs diarization, then Voxtral per segment, then KG extraction
# over every turn — minutes, not seconds, for anything longer than a short clip.
CONVERSATION_TIMEOUT = 3600
# Switching to Local blocks until Qwen3-14B is on the GPU: ~30 s from cache,
# ~10 min the first time the weights come down from HuggingFace.
MODE_SWITCH_TIMEOUT = 1800
RETRIEVE_TIMEOUT = 300
GENERATE_TIMEOUT = 600
STATUS_TIMEOUT = 10

MAX_ANSWER_TOKENS = 300

# What the visitor is asked to say, shown on the recording panel.
VISITOR_SCRIPT = """\
**Registra circa 15 secondi dicendo tre cose:**

1. Il tuo **nome**
2. Da **dove vieni**
3. Una cosa che ti piace, o un piccolo fatto su di te

*Esempio: «Ciao, sono Giulia, vengo da Napoli e sto imparando a stampare in 3D
i pezzi per la mia bicicletta.»*

*Vale anche per un file già registrato di **una sola persona**: caricalo qui e
dagli un nome.*
"""

# Enrollment modes shown in the Registrati tab.
MODE_SINGLE = "👤 Una persona (microfono o file)"
MODE_CONVERSATION = "👥 Più persone (conversazione registrata)"

CONVERSATION_SCRIPT = """\
**Carica una registrazione con più parlanti** (una riunione, una chiamata, un'intervista).

Il sistema separa i turni di parola e confronta ogni voce con quelle già in
memoria: chi è già stato registrato viene **riconosciuto automaticamente**, senza
scrivere nessun nome.

- Hai una trascrizione **.vtt** (Teams, Meet)? Caricala: nomi e tempi arrivano da
  lì e la diarizzazione viene saltata.
- Altrimenti ci pensa la diarizzazione.
"""


# ==========================================
# HTTP helpers — every call funnels through here so the booth always shows a
# readable message instead of a stack trace when the tunnel drops.
# ==========================================
class BackendError(RuntimeError):
    """Raised when the backend is unreachable or answers with a non-200."""


def _url(path: str) -> str:
    return f"{FASTAPI_URL}{path}"


def _request(method: str, path: str, timeout: int, **kwargs) -> Any:
    try:
        resp = requests.request(method, _url(path), timeout=timeout, **kwargs)
    except requests.exceptions.RequestException as e:
        raise BackendError(
            f"Backend non raggiungibile su {FASTAPI_URL}. "
            f"Controlla il port forwarding SSH. ({e.__class__.__name__})"
        ) from e
    if resp.status_code != 200:
        raise BackendError(f"Errore {resp.status_code} da {path}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as e:
        raise BackendError(f"Risposta non-JSON da {path}.") from e


def backend_mode() -> str:
    """Current KG-extractor mode ('local' / 'anthropic' / 'off'), 'unreachable' on failure."""
    try:
        return str(_request("GET", "/admin/extractor_mode", STATUS_TIMEOUT).get("mode", "off")).lower()
    except BackendError:
        return "unreachable"


def get_enrolled() -> Dict[str, Any]:
    return _request("GET", "/rag/enrolled", STATUS_TIMEOUT).get("enrolled_users", {}) or {}


def clip_counts() -> Dict[str, int]:
    """How many utterances each speaker has in the graph, keyed by user_id.

    `/rag/graph_viz` caps only the *entity* nodes, so every speaker is always
    returned with its `utterance_count`.
    """
    try:
        data = _request("GET", "/rag/graph_viz", STATUS_TIMEOUT, params={"max_entities": 1})
    except BackendError:
        return {}
    return {
        str(n["id"]): int(n.get("utterance_count", 0))
        for n in data.get("nodes", [])
        if n.get("type") == "speaker"
    }


def generate(prompt: str, max_new_tokens: int = MAX_ANSWER_TOKENS) -> str:
    """Answer a prompt with the backend's LLM (Qwen3-14B in Local mode)."""
    payload = {"prompt": prompt, "max_new_tokens": int(max_new_tokens)}
    return (_request("POST", "/llm/generate", GENERATE_TIMEOUT, json=payload).get("text") or "").strip()


def _upload(path: str, field: str, mime: str = "audio/wav") -> Tuple[str, Tuple[str, bytes, str]]:
    """One multipart part, read fully into memory.

    The basename is preserved because the backend writes the upload to a temp
    file under that name, and the *extension* is what lets ffmpeg/torchaudio
    decode it — the declared mime type is ignored server-side.
    """
    try:
        with open(path, "rb") as fh:
            return (field, (os.path.basename(path), fh.read(), mime))
    except OSError as e:
        raise BackendError(f"Impossibile leggere il file «{os.path.basename(path)}»: {e.strerror}.") from e


def _audio_files(paths: List[str], field: str = "audios") -> List[Tuple[str, Tuple[str, bytes, str]]]:
    """Read audio files into memory so no file handles are left open on the request."""
    return [_upload(path, field) for path in paths]


# ==========================================
# Fact-sheet parsing
#
# The backend returns the fact sheet as text. These patterns pull out the few
# fields the booth panels display; everything else stays in the raw view for
# the technical visitors.
# ==========================================
_RE_IDENTITY = re.compile(r"^\s*Identity:\s*(.+?)\s*$", re.M)
_RE_LANGUAGE = re.compile(r"^\s*Language:\s*(.+?)\s*$", re.M)
_RE_AGE = re.compile(r"^\s*Age Bracket:\s*(.+?)\s*$", re.M)
_RE_GENDER = re.compile(r"^\s*Gender:\s*(.+?)\s*$", re.M)
_RE_VOCAL = re.compile(r"^\s*-\s*Vocal Tone:\s*(.+?)\s*$", re.M)
_RE_SEMANTIC = re.compile(r"^\s*-\s*Semantic Tone:\s*(.+?)\s*$", re.M)
_RE_TRANSCRIPTION = re.compile(r'^\s*-\s*Transcription:\s*"(.*)"\s*$', re.M)

# Chip counters.
_RE_UTTERANCE = re.compile(r"\(Vocal:\s*.*?\|\s*Semantic:\s*.*?\):")
_RE_AUTHOR = re.compile(r"^Author:", re.M)
_RE_FACT_BLOCK = re.compile(
    r"Known typed facts about matched entities:\n((?:\s*-\s.*\n?|\s*\".*\"\n?)+)"
)
_RE_FACT_LINE = re.compile(r"^\s*-\s+\S", re.M)

UNKNOWN_PREFIX = "Unknown_Speaker"


def _first(pattern: re.Pattern, text: str, default: str = "") -> str:
    m = pattern.search(text or "")
    return m.group(1).strip() if m else default


def parse_probe(fact_sheet: str) -> Dict[str, str]:
    """Pull the per-probe attributes out of the AUDIO FILE RESOLUTION section."""
    identity = _first(_RE_IDENTITY, fact_sheet, "")
    return {
        "identity": identity,
        "known": bool(identity) and not identity.startswith(UNKNOWN_PREFIX),
        "transcription": _first(_RE_TRANSCRIPTION, fact_sheet, ""),
        "vocal": _first(_RE_VOCAL, fact_sheet, "?"),
        "semantic": _first(_RE_SEMANTIC, fact_sheet, "?"),
        "language": _first(_RE_LANGUAGE, fact_sheet, "?"),
        "age": _first(_RE_AGE, fact_sheet, "?"),
        "gender": _first(_RE_GENDER, fact_sheet, "?"),
    }


def count_evidence(fact_sheet: str) -> Dict[str, int]:
    """Count what the memory layer actually put on the table, for the chip row."""
    sheet = fact_sheet or ""
    # The probe's own utterance is rendered in the resolution section, not as
    # recalled evidence, so it must not be counted here.
    _, _, rest = sheet.partition("=== FACT SHEET: RELEVANT PAST STATEMENTS ===")
    body = rest or sheet
    facts = 0
    m = _RE_FACT_BLOCK.search(sheet)
    if m:
        facts = len(_RE_FACT_LINE.findall(m.group(1)))
    return {
        "statements": len(_RE_UTTERANCE.findall(body)) + len(_RE_AUTHOR.findall(body)),
        "facts": facts,
    }


# ==========================================
# HTML cards
# ==========================================
_CARD_CSS_CLASS = "booth-card"


def _card(title: str, body_html: str, accent: str = "#2563eb") -> str:
    return f"""
<div class="{_CARD_CSS_CLASS}" style="border-left:6px solid {accent};">
  <div class="booth-card-title" style="color:{accent};">{title}</div>
  {body_html}
</div>
"""


def _chip(label: str, value: str, color: str) -> str:
    return (
        f'<span class="booth-chip" style="background:{color}1a;color:{color};'
        f'border:1px solid {color}55;">{label}: <b>{value}</b></span>'
    )


def idle_card(message: str = "Registra un audio per iniziare.") -> str:
    return _card("🎙️ In attesa", f'<div class="booth-muted">{message}</div>', "#64748b")


def error_card(message: str) -> str:
    return _card("⚠️ Problema", f'<div class="booth-body">{message}</div>', "#dc2626")


def heard_card(probe: Dict[str, str]) -> str:
    """The 'this is what I heard' payoff card shown right after recording."""
    transcription = probe.get("transcription") or "(nessun parlato riconosciuto)"
    if probe.get("known"):
        headline = f'✅ Conosco questa voce: <b>{probe["identity"]}</b>'
        accent = "#16a34a"
        note = "Questa voce è già in memoria — puoi passare direttamente alle domande."
    else:
        headline = "🆕 Non conosco ancora questa voce"
        accent = "#2563eb"
        note = "Scrivi il nome qui sotto e aggiungila alla memoria."

    chips = (
        _chip("Tono di voce", probe.get("vocal", "?"), "#7c3aed")
        + _chip("Tono del testo", probe.get("semantic", "?"), "#0891b2")
        + _chip("Lingua", probe.get("language", "?"), "#64748b")
    )
    return _card(
        headline,
        f"""
        <div class="booth-quote">“{transcription}”</div>
        <div class="booth-chips">{chips}</div>
        <div class="booth-muted">{note}</div>
        """,
        accent,
    )


def enrolled_card(
    user_id: str, transcription: str, n_users: int, is_new: bool, clips: int
) -> str:
    quote = f'<div class="booth-quote">“{transcription}”</div>' if transcription else ""
    if is_new:
        title = f"🧠 <b>{user_id}</b> è entrato nella memoria"
        body = (
            f"La voce è ora un'identità stabile: la prossima volta il sistema "
            f"riconoscerà <b>{user_id}</b> dal solo suono, senza nome e senza login."
        )
    else:
        title = f"➕ Registrazione aggiunta al profilo di <b>{user_id}</b>"
        body = (
            f"Non è stata creata una nuova persona: questa voce si aggiunge a "
            f"<b>{user_id}</b>, che ora ha <b>{clips}</b> registrazioni in memoria. "
            f"Più campioni = riconoscimento più affidabile."
        )
    return _card(
        title,
        f"""
        {quote}
        <div class="booth-body">{body}</div>
        <div class="booth-muted">Persone in memoria: <b>{n_users}</b> ·
        Guarda il grafo nella scheda <b>🕸️ La memoria</b>.</div>
        """,
        "#16a34a",
    )


def evidence_chips(probe: Dict[str, str], evidence: Dict[str, int], used_audio: bool) -> str:
    chips = []
    if used_audio:
        if probe.get("known"):
            chips.append(_chip("Identità dalla voce", probe["identity"], "#16a34a"))
        else:
            chips.append(_chip("Identità dalla voce", "voce sconosciuta", "#dc2626"))
        if probe.get("vocal", "?") != "?":
            chips.append(_chip("Tono", probe["vocal"], "#7c3aed"))
    chips.append(_chip("Frasi richiamate", str(evidence["statements"]), "#2563eb"))
    chips.append(_chip("Fatti dal grafo", str(evidence["facts"]), "#c2410c"))
    return f'<div class="booth-chips">{"".join(chips)}</div>'


# ==========================================
# Tab 1 — Registrati (enrollment)
# ==========================================
_LISTEN_QUESTION = "Who is the speaker in this audio and what have they told us before?"

_NAME_STRIP = re.compile(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ]+")


def sanitize_name(raw: str) -> str:
    """Reduce a typed name to a single safe token.

    The name becomes the `user_id` used as a graph node and matched against
    questions with a word-boundary regex, so spaces and punctuation are
    stripped and the result is capitalised for display.
    """
    first = (raw or "").strip().split()[0] if (raw or "").strip() else ""
    return _NAME_STRIP.sub("", first)[:24].capitalize()


def resolve_user_id(name: str, existing: Dict[str, Any], distinct_person: bool) -> Tuple[str, bool]:
    """Map a typed name to the user_id to enroll under.

    Returns (user_id, is_new_person).

    Re-using a name is the normal case: several clips of the same person make a
    stronger acoustic profile, and the backend stores one embedding per clip
    under the shared user_id. So by default the name is taken literally and the
    recording is appended to that person's profile.

    `distinct_person` is the escape hatch for a genuine collision — a *different*
    visitor who happens to be called Marco. Only then is the name suffixed, so
    the two never share one speaker node or one voice profile.
    """
    taken = {str(k).lower(): str(k) for k in existing}
    if name.lower() not in taken:
        return name, True
    if not distinct_person:
        return taken[name.lower()], False
    i = 2
    while f"{name}_{i}".lower() in taken:
        i += 1
    return f"{name}_{i}", True


def listen(audio_path: Optional[str]):
    """Transcribe the clip and report whether the voice is already known.

    Runs the recording through `/rag/retrieve` as a probe. The backend deletes
    the temporary probe embedding afterwards, so this does not pollute the
    gallery — it is a read-only look at the audio.
    """
    if not audio_path:
        return idle_card("Nessun audio: premi il microfono e registra."), "", ""

    try:
        result = _request(
            "POST",
            "/rag/retrieve",
            RETRIEVE_TIMEOUT,
            files=_audio_files([audio_path]),
            data={"question": _LISTEN_QUESTION},
        )
    except BackendError as e:
        return error_card(str(e)), "", ""

    fact_sheet = result.get("fact_sheet", "")
    probe = parse_probe(fact_sheet)
    return heard_card(probe), fact_sheet, probe.get("transcription", "")


def enroll(
    audio_path: Optional[str],
    raw_name: str,
    heard_transcription: str,
    distinct_person: bool = False,
):
    """Enroll the recorded clip under the typed name.

    Re-typing an existing name adds the clip to that person's profile. Tick
    `distinct_person` only when this really is someone else with the same name.
    """
    if not audio_path:
        return error_card("Manca l'audio: registra prima di salvare."), gr.update()

    name = sanitize_name(raw_name)
    if len(name) < 2:
        return error_card("Scrivi un nome valido (almeno 2 lettere, senza spazi)."), gr.update()

    try:
        user_id, is_new = resolve_user_id(name, get_enrolled(), bool(distinct_person))
        _request(
            "POST",
            "/rag/index",
            ENROLL_TIMEOUT,
            files=_audio_files([audio_path]),
            data={"user_ids": [user_id]},
        )
        names = sorted(get_enrolled().keys())
        clips = clip_counts().get(user_id, 0)
    except BackendError as e:
        return error_card(str(e)), gr.update()

    card = enrolled_card(user_id, (heard_transcription or "").strip(), len(names), is_new, clips)
    return card, gr.update(choices=names, value=user_id)


UNKNOWN_CLUSTER_PREFIX = "unknown_"


def conversation_card(result: Dict[str, Any], source: str) -> str:
    """Render the outcome of a multi-speaker enrollment.

    `identity_map` maps each diarization cluster to a resolved identity. When a
    cluster centroid matched an already-enrolled voice the value is that
    person's user_id — the cross-recording recognition the system is built for.
    Otherwise it is `unknown_<cluster>`, a new participant.
    """
    identity_map = result.get("identity_map", {}) or {}
    rows = []
    recognised = 0
    for cluster, identity in sorted(identity_map.items()):
        if str(identity).startswith(UNKNOWN_CLUSTER_PREFIX):
            rows.append(
                f'<li><code>{cluster}</code> → <b>{identity}</b> '
                f'<span style="color:#64748b">(nuovo interlocutore)</span></li>'
            )
        else:
            recognised += 1
            rows.append(
                f'<li><code>{cluster}</code> → <b>{identity}</b> '
                f'<span style="color:#16a34a">🎯 riconosciuto dalla memoria</span></li>'
            )

    headline = (
        f"🎯 Conversazione registrata — {recognised} voce/i già nota/e"
        if recognised
        else "👥 Conversazione registrata"
    )
    note = (
        "Le persone che hanno parlato in questa conversazione sono ora collegate "
        "fra loro nel grafo (<b>si conoscono</b>)."
    )
    return _card(
        headline,
        f"""
        <div class="booth-body">
          Segmenti aggiunti alla memoria: <b>{result.get('enrolled_segments', 0)}</b><br>
          Conversazione: <code>{result.get('dialogue_id', '?')}</code> · fonte: {source}
        </div>
        <div class="booth-body"><b>Chi ha parlato</b>
          <ul style="margin:6px 0 0 18px">{''.join(rows) or '<li>nessun interlocutore rilevato</li>'}</ul>
        </div>
        <div class="booth-muted">{note}</div>
        """,
        "#16a34a",
    )


def enroll_conversation(
    audio_path: Optional[str],
    vtt_path: Optional[str],
    num_speakers: Optional[float],
    speaker_map_json: str,
):
    """Enroll a multi-speaker recording through `/rag/index_conversation`.

    Three backend paths, in the order the endpoint prefers them:
      - a VTT transcript wins outright: real speaker names and timestamps come
        from the file and pyannote is skipped;
      - otherwise a speaker map, if given, names the pyannote clusters;
      - otherwise full diarization + clustering, with each cluster centroid
        matched against the enrolled voices.
    """
    if not audio_path:
        return error_card("Carica prima il file audio della conversazione."), gr.update()

    # Validate cheap inputs before reading a possibly large audio file into memory.
    data: Dict[str, Any] = {}
    source = "diarizzazione automatica"
    if vtt_path:
        source = "trascrizione VTT"
    else:
        # Both are ignored by the backend when a VTT is present, so only send
        # them on the diarization path to keep the request honest.
        cleaned_map = (speaker_map_json or "").strip()
        if cleaned_map:
            try:
                parsed = json.loads(cleaned_map)
            except json.JSONDecodeError:
                return (
                    error_card(
                        "La mappa dei parlanti non è JSON valido. "
                        'Esempio: {"SPEAKER_00": "Abdul", "SPEAKER_01": "Ian"}'
                    ),
                    gr.update(),
                )
            if not isinstance(parsed, dict):
                return error_card("La mappa dei parlanti deve essere un oggetto JSON."), gr.update()
            data["speaker_map"] = cleaned_map
            source = "mappa dei parlanti"
        if num_speakers and int(num_speakers) >= 2:
            data["num_speakers"] = int(num_speakers)

    try:
        files = [_upload(audio_path, "audio")]
        if vtt_path:
            files.append(_upload(vtt_path, "vtt_file", "text/vtt"))
        result = _request(
            "POST", "/rag/index_conversation", CONVERSATION_TIMEOUT, files=files, data=data
        )
        names = sorted(get_enrolled().keys())
    except BackendError as e:
        return error_card(str(e)), gr.update()

    if result.get("enrolled_segments", 0) == 0:
        return (
            error_card(
                "Nessun segmento di parlato trovato nel file. "
                "Controlla che l'audio contenga voce e sia leggibile."
            ),
            gr.update(choices=names),
        )

    return conversation_card(result, source), gr.update(choices=names)


def switch_enroll_mode(mode: str):
    """Show one enrollment panel at a time and clear the previous result."""
    single = mode == MODE_SINGLE
    placeholder = (
        "Registra un audio per iniziare."
        if single
        else "Carica una conversazione per iniziare."
    )
    return (
        gr.update(visible=single),
        gr.update(visible=not single),
        idle_card(placeholder),
        "",
    )


def seed_enroll(paths: Optional[List[str]], names_csv: str):
    """Bulk-enroll prepared recordings — one name per file, in order.

    Used before the doors open to seed the memory with a handful of colleagues
    talking about the same topic, so the first visitor already has a populated
    graph to query.

    Names are taken literally: repeating a name enrolls several clips under one
    person, which is how you give a colleague a strong voice profile. Use
    distinct names when you mean distinct people.
    """
    if not paths:
        return "Nessun file selezionato.", gr.update()

    raw_names = [n.strip() for n in (names_csv or "").split(",") if n.strip()]
    if len(raw_names) != len(paths):
        return (
            f"Servono {len(paths)} nomi separati da virgola (ne hai dati {len(raw_names)}).",
            gr.update(),
        )

    log: List[str] = []
    try:
        roster = get_enrolled()
    except BackendError as e:
        return str(e), gr.update()

    for path, raw in zip(paths, raw_names):
        name = sanitize_name(raw)
        if len(name) < 2:
            log.append(f"✗ {raw}: nome non valido, saltato.")
            continue
        # Literal: a repeated name deliberately adds another clip to that person.
        user_id = name
        try:
            _request(
                "POST",
                "/rag/index",
                ENROLL_TIMEOUT,
                files=_audio_files([path]),
                data={"user_ids": [user_id]},
            )
            roster[user_id] = {}
            log.append(f"✓ {os.path.basename(path)} → {user_id}")
        except BackendError as e:
            log.append(f"✗ {os.path.basename(path)}: {e}")

    try:
        names = sorted(get_enrolled().keys())
    except BackendError:
        names = sorted(roster.keys())

    log.append("")
    log.append(f"Persone in memoria: {len(names)}")
    return "\n".join(log), gr.update(choices=names)


# ==========================================
# Tab 2 — Chiedi (query with optional probe audio)
#
# Preset buttons show an Italian label but send an English question string.
# The backend's question router (`_classify_question` in rag.py) matches
# English patterns only, so this keeps the aggregate / participant branches
# reachable while the booth stays in Italian.
# ==========================================
PRESETS: List[Tuple[str, str]] = [
    ("🎤 Chi sta parlando?", "Who is the speaker in this audio? State their identity."),
    ("💬 Cosa mi ha raccontato?", "What has {name} told us? Summarise everything {name} said."),
    ("📍 Di dove è?", "Where is {name} from?"),
    ("🔎 Chi ha parlato di...?", "Which speakers mentioned {topic}? List them."),
    ("😊 Chi era più contento?", "Which speaker expresses joy most often?"),
    ("🔢 Quante volte gioia?", "How many utterances by {name} express joy?"),
]

_NO_MEMORY_PROMPT = """You are a helpful assistant answering a visitor at a science fair.
You have no stored memory of any person and no audio input.
Answer the question below. If you do not have the information, say so plainly and briefly.

### Question:
{question}

### Answer:"""

_LANG_INSTRUCTION = {
    "Italiano": "IMPORTANT: write your final answer in Italian, in at most 3 short sentences.",
    "English": "IMPORTANT: write your final answer in English, in at most 3 short sentences.",
}


def fill_preset(template: str, name: Optional[str], topic: str) -> str:
    """Render a preset into a concrete question."""
    filled = template
    if "{name}" in filled:
        filled = filled.replace("{name}", (name or "the speaker").strip() or "the speaker")
    if "{topic}" in filled:
        filled = filled.replace("{topic}", (topic or "").strip() or "that topic")
    return filled


def _with_language(prompt: str, language: str) -> str:
    """Insert the output-language instruction just before the answer marker.

    The backend builds the prompt and ends it with '### Answer:', so the
    instruction has to go above that line to still be read as an instruction.
    """
    instruction = _LANG_INSTRUCTION.get(language)
    if not instruction:
        return prompt
    marker = "### Answer:"
    idx = prompt.rfind(marker)
    if idx == -1:
        return f"{prompt}\n\n{instruction}"
    return f"{prompt[:idx]}{instruction}\n\n{prompt[idx:]}"


def ask(
    probe_path: Optional[str],
    question: str,
    language: str,
    show_baseline: bool,
    history: Optional[List[Dict[str, str]]],
):
    """Answer one question twice: once bare, once with the memory fact sheet.

    Both answers come from the same backend LLM through `/llm/generate`; the
    only difference is the prompt, so the comparison isolates the memory layer.
    """
    history = history or []
    question = (question or "").strip()
    if not question:
        return (
            idle_card("Scrivi o scegli una domanda."),
            "",
            "",
            "",
            history,
            format_history(history),
        )

    files = _audio_files([probe_path]) if probe_path else None
    data: Dict[str, Any] = {"question": question}
    if history:
        data["conversation_history"] = json.dumps(history)

    try:
        result = _request("POST", "/rag/retrieve", RETRIEVE_TIMEOUT, files=files, data=data)
    except BackendError as e:
        return error_card(str(e)), "", "", "", history, format_history(history)

    fact_sheet = result.get("fact_sheet", "")
    recommended = result.get("recommended_prompt", "")
    if not recommended:
        return (
            error_card("Il backend non ha restituito un prompt."),
            "",
            "",
            fact_sheet,
            history,
            format_history(history),
        )

    probe = parse_probe(fact_sheet) if probe_path else {}
    chips = evidence_chips(probe, count_evidence(fact_sheet), used_audio=bool(probe_path))

    try:
        memory_answer = generate(_with_language(recommended, language))
    except BackendError as e:
        return error_card(str(e)), "", chips, fact_sheet, history, format_history(history)

    baseline_answer = ""
    if show_baseline:
        try:
            baseline_answer = generate(
                _with_language(_NO_MEMORY_PROMPT.format(question=question), language)
            )
        except BackendError as e:
            baseline_answer = f"(baseline non disponibile: {e})"

    new_history = history + [{"question": question, "answer": memory_answer}]
    return (
        baseline_answer or "(confronto disattivato)",
        memory_answer,
        chips,
        fact_sheet,
        new_history,
        format_history(new_history),
    )


def format_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return "(nessuna domanda precedente)"
    lines = []
    for i, turn in enumerate(history, start=1):
        lines.append(f"{i}. D: {turn.get('question', '')}")
        lines.append(f"   R: {turn.get('answer', '')}")
    return "\n".join(lines)


def reset_history():
    return [], format_history([])


# ==========================================
# Tab 3 — La memoria (knowledge graph)
# ==========================================
_ENTITY_COLORS = {
    "PERSON": "#2ecc71",
    "LOCATION": "#e74c3c",
    "ORGANIZATION": "#e67e22",
    "EVENT": "#1abc9c",
    "OBJECT": "#9b59b6",
    "CONCEPT": "#3498db",
    "WORK": "#f39c12",
    "GPE": "#e74c3c",
    "ORG": "#e67e22",
    "PRODUCT": "#9b59b6",
    "WORK_OF_ART": "#f39c12",
    "FAC": "#d35400",
    "UNKNOWN": "#95a5a6",
}

_EMPTY_GRAPH_HTML = """
<div style="display:flex;align-items:center;justify-content:center;height:640px;
            background:#f8fafc;border-radius:12px;border:2px dashed #cbd5e1;">
  <div style="text-align:center;color:#64748b;">
    <div style="font-size:64px;margin-bottom:16px;">🕸️</div>
    <div style="font-size:20px;font-weight:700;">La memoria è ancora vuota</div>
    <div style="font-size:15px;margin-top:8px;">Registra la prima persona nella scheda «Registrati».</div>
  </div>
</div>
"""


def graph_view(max_entities: int = 40):
    """Render the persistent memory as an interactive graph (second screen)."""
    try:
        data = _request(
            "GET", "/rag/graph_viz", 60, params={"max_entities": int(max_entities)}
        )
    except BackendError as e:
        return f"<p style='color:#dc2626;padding:16px'>{e}</p>", ""

    stats = data.get("stats", {})
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    stats_str = (
        f"👤 Persone: {stats.get('speaker', 0)}   |   "
        f"🏷️ Concetti: {stats.get('entity', 0)}   |   "
        f"💬 Frasi in memoria: {stats.get('utterance', 0)}   |   "
        f"🔗 Collegamenti: {stats.get('edges', 0)}"
    )

    if not nodes:
        return _EMPTY_GRAPH_HTML, stats_str

    net = Network(
        height="660px", width="100%", bgcolor="#f8fafc",
        font_color="#1e293b", cdn_resources="in_line",
    )

    for n in nodes:
        nid = n["id"]
        if n["type"] == "speaker":
            net.add_node(
                nid,
                label=str(nid).capitalize(),
                shape="box",
                color={"background": "#2563eb", "border": "#1d4ed8",
                       "highlight": {"background": "#60a5fa", "border": "#2563eb"}},
                font={"color": "#ffffff", "size": 18, "bold": True},
                size=34,
                title=f"<b>{str(nid).capitalize()}</b><br>Frasi in memoria: {n.get('utterance_count', 0)}",
                borderWidth=3,
            )
        else:
            label = n.get("label", "UNKNOWN")
            surface = n.get("surface", nid)
            mentions = n.get("total_mentions", 1)
            color = _ENTITY_COLORS.get(label, "#95a5a6")
            tooltip = f"<b>{surface}</b><br>Tipo: {label}<br>Citato {mentions}×"
            if n.get("description"):
                tooltip += f"<br><i>{n['description'][:300]}</i>"
            net.add_node(
                nid,
                label=str(surface)[:24],
                shape="dot",
                color={"background": color, "border": color,
                       "highlight": {"background": color, "border": "#1e293b"}},
                size=min(12 + mentions * 3, 34),
                title=tooltip,
                font={"size": 14},
            )

    for e in edges:
        if e.get("is_typed_fact"):
            rel = e.get("relation", "FACT")
            tooltip = f"<b>{rel}</b>"
            if e.get("fact"):
                tooltip += f"<br>{e['fact']}"
            net.add_edge(
                e["source"], e["target"], label=rel,
                color={"color": "#7c3aed", "highlight": "#a78bfa"},
                font={"size": 12, "color": "#5b21b6", "strokeWidth": 3, "strokeColor": "#ffffff"},
                width=2.5, title=tooltip,
                arrows={"to": {"enabled": True, "scaleFactor": 0.7}},
            )
        elif e.get("relation") == "KNOWS":
            n_dial = e.get("n_dialogues", 0)
            net.add_edge(
                e["source"], e["target"], label="SI CONOSCONO",
                color={"color": "#1e3a5f", "highlight": "#2563eb"},
                font={"size": 12, "color": "#1e3a5f", "strokeWidth": 3, "strokeColor": "#ffffff"},
                width=2 + min(n_dial, 5),
                title=f"Hanno parlato nella stessa conversazione ({n_dial})",
            )
        else:
            cnt = e.get("count", 1)
            net.add_edge(
                e["source"], e["target"], label=f"parla di ({cnt}×)",
                color={"color": "#94a3b8", "highlight": "#64748b"},
                font={"size": 11, "color": "#64748b", "strokeWidth": 3, "strokeColor": "#ffffff"},
                width=max(0.5, min(cnt * 0.6, 3.0)), title=f"MENTIONS — {cnt}×", dashes=True,
            )

    net.set_options("""{
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -70,
          "springLength": 200,
          "springConstant": 0.08,
          "damping": 0.4
        },
        "stabilization": {"iterations": 200}
      },
      "interaction": {
        "hover": true, "tooltipDelay": 100, "zoomView": true,
        "dragView": true, "dragNodes": true, "navigationButtons": true, "keyboard": true
      },
      "edges": {"smooth": {"type": "continuous"}, "font": {"align": "middle"}}
    }""")

    b64 = base64.b64encode(net.generate_html().encode()).decode()
    iframe = (
        f'<iframe src="data:text/html;base64,{b64}" width="100%" height="680px" '
        f'style="border:none;border-radius:12px;"></iframe>'
    )
    return iframe, stats_str


# ==========================================
# Tab 4 — Regia (operator controls)
# ==========================================
def status_line() -> str:
    mode = backend_mode()
    if mode == "unreachable":
        return f"🔴 Backend non raggiungibile su {FASTAPI_URL} — controlla il tunnel SSH."
    try:
        n_users = len(get_enrolled())
    except BackendError:
        n_users = 0
    if mode == "local":
        return f"🟢 Backend attivo (Qwen3-14B locale) · {n_users} persone in memoria."
    if mode == "anthropic":
        return (
            f"🟡 Backend in modalità Claude API · {n_users} persone in memoria. "
            "Le risposte richiedono /llm/generate, disponibile solo in modalità Local."
        )
    return (
        f"🟡 Estrattore KG disattivato · {n_users} persone in memoria. "
        "Senza modalità Local il grafo non cresce e /llm/generate non risponde."
    )


def activate_local_mode() -> str:
    """Load Qwen3-14B on the backend and route KG extraction through it.

    Everything the booth needs depends on this: `/llm/generate` returns 409 in
    any other mode (so both answer panels stay empty), and `persist()` skips
    entity extraction when no extractor is available (so the graph never grows
    past bare speaker nodes).

    The backend blocks until the weights are on the GPU — ~30 s from cache,
    ~10 min the first time it has to pull them from HuggingFace.
    """
    try:
        _request("POST", "/admin/extractor_mode", MODE_SWITCH_TIMEOUT, json={"mode": "local"})
    except BackendError as e:
        return f"🔴 Attivazione fallita.\n{e}"
    return status_line()


def consolidate_memory() -> str:
    """Flush queued KG extraction and dedupe entities, without enrolling anything.

    Utterances ingested while the extractor was off are flagged
    `entities_pending` and stay that way until some call persists the graph.
    `/admin/dedupe_entities` persists at the end of its pass, so it doubles as a
    way to work through that backlog before the doors open, rather than having
    it land on the first visitor's enrollment.
    """
    try:
        stats = _request("POST", "/admin/dedupe_entities", CONVERSATION_TIMEOUT, json={})
    except BackendError as e:
        return f"🔴 Consolidamento fallito.\n{e}"
    lines = [
        "✅ Consolidamento completato.",
        f"  entità esaminate: {stats.get('candidates_examined', 0)}",
        f"  entità unite:     {stats.get('entities_merged', 0)}",
        "",
        "L'estrazione arretrata è stata eseguita durante il salvataggio del grafo,",
        "ma non viene riportata qui: controlla la scheda 🕸️ La memoria.",
    ]
    return "\n".join(lines)


def roster_text() -> str:
    """List the people in memory with how many clips each one has.

    The clip count is the useful number: `/rag/enrolled` de-duplicates by
    filename, and every mic recording arrives as `0_audio.wav`, so the filename
    column cannot show that a second clip landed on a profile.
    """
    try:
        users = get_enrolled()
    except BackendError as e:
        return str(e)
    if not users:
        return "Nessuna persona in memoria."
    counts = clip_counts()
    lines = [f"Persone in memoria: {len(users)}", "-" * 40]
    for uid, attrs in sorted(users.items()):
        n = counts.get(uid, 0)
        suffix = "registrazione" if n == 1 else "registrazioni"
        lines.append(f"👤 {uid}   ({attrs.get('language', '?')})   ·   {n} {suffix}")
    return "\n".join(lines)


def refresh_names():
    try:
        names = sorted(get_enrolled().keys())
    except BackendError:
        names = []
    return gr.update(choices=names)


def clear_database(confirmed: bool):
    """Wipe the whole store. There is no per-person delete in the API, so a
    visitor withdrawing consent means resetting the booth — do it between
    sessions, or at the end of the day."""
    if not confirmed:
        return "Spunta la casella di conferma prima di cancellare.", gr.update()
    try:
        _request("DELETE", "/rag/clear", 120)
    except BackendError as e:
        return str(e), gr.update()
    return "Memoria cancellata. Il grafo riparte da zero.", gr.update(choices=[], value=None)


# ==========================================
# Styling
# ==========================================
BOOTH_CSS = """
.booth-card {
  background: var(--background-fill-primary);
  border-radius: 12px;
  padding: 18px 22px;
  margin: 4px 0 10px 0;
  box-shadow: 0 1px 3px rgba(15,23,42,.10);
}
.booth-card-title { font-size: 20px; font-weight: 700; margin-bottom: 10px; }
.booth-quote {
  font-size: 19px; line-height: 1.5; font-style: italic;
  margin: 10px 0; color: var(--body-text-color);
}
.booth-body { font-size: 16px; line-height: 1.55; }
.booth-muted { font-size: 14px; color: var(--body-text-color-subdued); margin-top: 8px; }
.booth-chips { margin: 12px 0 4px 0; display: flex; flex-wrap: wrap; gap: 8px; }
.booth-chip {
  display: inline-block; padding: 5px 12px; border-radius: 999px;
  font-size: 14px; white-space: nowrap;
}
#booth-answer-memory textarea { font-size: 17px !important; line-height: 1.55 !important; }
#booth-answer-baseline textarea { font-size: 16px !important; color: #64748b !important; }
"""


# ==========================================
# Layout
# ==========================================
with gr.Blocks(title="SpeakerRAGent · Maker Faire Rome", fill_width=True) as demo:
    gr.Markdown(
        "# 🧠 SpeakerRAGent — la memoria che riconosce la voce\n"
        "*Fondazione Bruno Kessler · Maker Faire Rome*"
    )
    status_box = gr.Markdown(value="")

    heard_state = gr.State("")
    history_state = gr.State([])

    with gr.Tabs():

        # ---------- Tab 1: enrollment ----------
        with gr.Tab("🎙️ Registrati"):
            enroll_mode = gr.Radio(
                choices=[MODE_SINGLE, MODE_CONVERSATION],
                value=MODE_SINGLE,
                label="Cosa stai registrando?",
            )
            with gr.Row():
                with gr.Column(scale=1):

                    # --- Mode A: one speaker, from the mic or from a file ---
                    with gr.Group(visible=True) as single_group:
                        gr.Markdown(VISITOR_SCRIPT)
                        enroll_audio = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            format="wav",
                            label="Registrazione dal microfono, oppure carica un file",
                        )
                        listen_btn = gr.Button("🔍 1. Ascolta", variant="secondary", size="lg")
                        enroll_name = gr.Textbox(
                            label="Nome (una parola sola)",
                            placeholder="Giulia",
                            info="Nome già in memoria = stessa persona: la registrazione "
                                 "si aggiunge al suo profilo vocale.",
                            max_lines=1,
                        )
                        enroll_distinct = gr.Checkbox(
                            value=False,
                            label="È un'altra persona con lo stesso nome",
                            info="Spunta solo per un omonimo: verrà creato Nome_2.",
                        )
                        enroll_btn = gr.Button(
                            "🧠 2. Aggiungimi alla memoria", variant="primary", size="lg"
                        )

                    # --- Mode B: a multi-speaker recording ---
                    with gr.Group(visible=False) as conversation_group:
                        gr.Markdown(CONVERSATION_SCRIPT)
                        conv_audio = gr.File(
                            label="File audio della conversazione",
                            file_count="single",
                            type="filepath",
                        )
                        conv_vtt = gr.File(
                            label="Trascrizione .vtt (facoltativa — es. da Teams)",
                            file_count="single",
                            type="filepath",
                        )
                        with gr.Accordion("Opzioni per la diarizzazione", open=False):
                            gr.Markdown(
                                "Usate **solo** quando non carichi un file VTT: con il VTT "
                                "i nomi e i tempi arrivano già dalla trascrizione."
                            )
                            conv_num_speakers = gr.Number(
                                label="Quanti parlanti (facoltativo)",
                                value=None,
                                precision=0,
                                info="Lascia vuoto per farlo stimare al sistema.",
                            )
                            conv_speaker_map = gr.Textbox(
                                label="Nomi dei parlanti (JSON, facoltativo)",
                                placeholder='{"SPEAKER_00": "Abdul", "SPEAKER_01": "Ian"}',
                                max_lines=3,
                            )
                        conv_btn = gr.Button(
                            "👥 Aggiungi la conversazione alla memoria",
                            variant="primary",
                            size="lg",
                        )

                with gr.Column(scale=1):
                    enroll_card = gr.HTML(value=idle_card())
                    with gr.Accordion("🔬 Fact sheet (vista tecnica)", open=False):
                        enroll_sheet = gr.Textbox(label="", lines=16, buttons=["copy"])

            enroll_mode.change(
                fn=switch_enroll_mode,
                inputs=[enroll_mode],
                outputs=[single_group, conversation_group, enroll_card, enroll_sheet],
            )
            listen_btn.click(
                fn=listen,
                inputs=[enroll_audio],
                outputs=[enroll_card, enroll_sheet, heard_state],
            )

        # ---------- Tab 2: query ----------
        with gr.Tab("💬 Chiedi"):
            gr.Markdown(
                "Registra una **nuova** frase: viene usata come *probe*, cioè la voce con cui "
                "il sistema capisce chi sta parlando. Poi scegli una domanda."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    probe_audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        format="wav",
                        label="Registrazione di prova (facoltativa)",
                    )
                    with gr.Row():
                        speaker_pick = gr.Dropdown(
                            label="Persona", choices=[], value=None,
                            allow_custom_value=True, scale=1,
                        )
                        topic_box = gr.Textbox(
                            label="Argomento", placeholder="Napoli, robot, stampa 3D…",
                            max_lines=1, scale=1,
                        )
                    gr.Markdown("**Domande pronte**")
                    preset_buttons = []
                    for row_start in range(0, len(PRESETS), 2):
                        with gr.Row():
                            for label, template in PRESETS[row_start:row_start + 2]:
                                preset_buttons.append((gr.Button(label, size="sm"), template))
                    question_box = gr.Textbox(
                        label="Domanda", lines=2,
                        placeholder="Who is the speaker in this audio?",
                    )
                    with gr.Row():
                        language_pick = gr.Radio(
                            choices=["Italiano", "English"], value="Italiano",
                            label="Lingua della risposta", scale=2,
                        )
                        baseline_toggle = gr.Checkbox(
                            value=True, label="Mostra il confronto senza memoria", scale=1,
                        )
                    ask_btn = gr.Button("💬 Chiedi", variant="primary", size="lg")

                with gr.Column(scale=2):
                    chips_html = gr.HTML(value="")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### ❌ LLM senza memoria\n*Stesso modello, nessun contesto — e non può sentire la voce.*")
                            baseline_answer = gr.Textbox(
                                label="", lines=8, elem_id="booth-answer-baseline",
                                buttons=["copy"],
                            )
                        with gr.Column():
                            gr.Markdown("### ✅ LLM + SpeakerRAGent\n*Stesso modello, con la memoria ancorata alla voce.*")
                            memory_answer = gr.Textbox(
                                label="", lines=8, elem_id="booth-answer-memory",
                                buttons=["copy"],
                            )
                    with gr.Accordion("🔬 Fact sheet (vista tecnica)", open=False):
                        ask_sheet = gr.Textbox(label="", lines=18, buttons=["copy"])
                    with gr.Accordion("🗂️ Domande precedenti", open=False):
                        history_box = gr.Textbox(
                            label="", value=format_history([]), lines=6, interactive=False
                        )
                        reset_btn = gr.Button("🧹 Azzera la conversazione", variant="stop", size="sm")

            for btn, template in preset_buttons:
                btn.click(
                    fn=fill_preset,
                    inputs=[gr.State(template), speaker_pick, topic_box],
                    outputs=[question_box],
                )

            ask_btn.click(
                fn=ask,
                inputs=[probe_audio, question_box, language_pick, baseline_toggle, history_state],
                outputs=[
                    baseline_answer, memory_answer, chips_html,
                    ask_sheet, history_state, history_box,
                ],
            )
            reset_btn.click(fn=reset_history, inputs=[], outputs=[history_state, history_box])

        # ---------- Tab 3: graph ----------
        with gr.Tab("🕸️ La memoria"):
            gr.Markdown(
                "Ogni **riquadro blu** è una persona, ogni **pallino** una cosa di cui ha parlato. "
                "Il grafo cresce durante la giornata."
            )
            with gr.Row():
                graph_max = gr.Slider(
                    label="Concetti mostrati", minimum=5, maximum=120, value=40, step=5, scale=3
                )
                graph_btn = gr.Button("🔄 Aggiorna", variant="primary", scale=1)
                graph_auto = gr.Checkbox(label="Auto (30 s)", value=False, scale=1)
            graph_stats = gr.Markdown(value="")
            graph_html = gr.HTML(value=_EMPTY_GRAPH_HTML)
            graph_timer = gr.Timer(30, active=False)

            graph_btn.click(fn=graph_view, inputs=[graph_max], outputs=[graph_html, graph_stats])
            graph_timer.tick(fn=graph_view, inputs=[graph_max], outputs=[graph_html, graph_stats])
            graph_auto.change(
                fn=lambda on: gr.Timer(30, active=bool(on)), inputs=[graph_auto], outputs=[graph_timer]
            )

        # ---------- Tab 4: operator ----------
        with gr.Tab("⚙️ Regia"):
            gr.Markdown(
                "Controlli per chi presenta. Il backend deve essere in **modalità Local** "
                "(Qwen3-14B) perché `/llm/generate` risponda e il grafo si popoli."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Stato")
                    admin_status = gr.Textbox(label="", lines=3, interactive=False)
                    with gr.Row():
                        admin_refresh = gr.Button("🔄 Aggiorna stato", variant="secondary")
                        admin_local_btn = gr.Button(
                            "🟢 Attiva Qwen3-14B locale", variant="primary"
                        )
                    gr.Markdown(
                        "Senza la modalità Local il backend non risponde a `/llm/generate` "
                        "(le due risposte restano vuote) e non estrae entità, quindi il grafo "
                        "resta fermo ai soli nodi delle persone. L'attivazione carica i pesi "
                        "sulla GPU: ~30 s se già in cache."
                    )
                    admin_consolidate_btn = gr.Button(
                        "🧩 Consolida la memoria (recupera l'arretrato)", variant="secondary"
                    )
                    gr.Markdown(
                        "Le registrazioni fatte con l'estrattore spento restano in coda. "
                        "Premi qui **dopo** l'attivazione per smaltirle tutte insieme, invece "
                        "di farle ricadere sulla prima registrazione di un visitatore."
                    )
                    admin_consolidate_log = gr.Textbox(label="", lines=7, interactive=False)
                    gr.Markdown("### Persone in memoria")
                    admin_roster = gr.Textbox(label="", lines=14, interactive=False)
                    roster_btn = gr.Button("👥 Mostra elenco", variant="secondary")
                with gr.Column():
                    gr.Markdown(
                        "### Preparazione: colleghi già registrati\n"
                        "Carica un file per persona e scrivi i nomi nello stesso ordine, "
                        "separati da virgola. Registrali mentre parlano **dello stesso tema**, "
                        "così il grafo avrà già dei concetti condivisi all'apertura."
                    )
                    seed_files = gr.File(
                        label="File audio (uno per persona)",
                        file_count="multiple",
                        type="filepath",
                    )
                    seed_names = gr.Textbox(
                        label="Nomi in ordine", placeholder="Alessio, Marco, Sara, Luca"
                    )
                    seed_btn = gr.Button("📥 Registra i colleghi", variant="primary")
                    seed_log = gr.Textbox(label="Esito", lines=10, interactive=False)

                    gr.Markdown(
                        "### Reset\n"
                        "L'API non espone la cancellazione di una singola persona: per una "
                        "richiesta di cancellazione si azzera tutto. Fallo a fine giornata."
                    )
                    clear_confirm = gr.Checkbox(label="Confermo: cancella tutta la memoria", value=False)
                    clear_btn = gr.Button("🗑️ Cancella tutto", variant="stop")
                    clear_log = gr.Textbox(label="", lines=2, interactive=False)

            admin_refresh.click(fn=status_line, inputs=[], outputs=[admin_status])
            admin_local_btn.click(
                fn=activate_local_mode, inputs=[], outputs=[admin_status]
            ).then(fn=status_line, inputs=[], outputs=[status_box])
            admin_consolidate_btn.click(
                fn=consolidate_memory, inputs=[], outputs=[admin_consolidate_log]
            )
            roster_btn.click(fn=roster_text, inputs=[], outputs=[admin_roster])
            seed_btn.click(
                fn=seed_enroll,
                inputs=[seed_files, seed_names],
                outputs=[seed_log, speaker_pick],
            )
            clear_btn.click(
                fn=clear_database,
                inputs=[clear_confirm],
                outputs=[clear_log, speaker_pick],
            )

    # Enrollment refreshes the speaker picker used by the preset questions.
    enroll_btn.click(
        fn=enroll,
        inputs=[enroll_audio, enroll_name, heard_state, enroll_distinct],
        outputs=[enroll_card, speaker_pick],
    )
    conv_btn.click(
        fn=enroll_conversation,
        inputs=[conv_audio, conv_vtt, conv_num_speakers, conv_speaker_map],
        outputs=[enroll_card, speaker_pick],
    )

    demo.load(fn=status_line, inputs=[], outputs=[status_box])
    demo.load(fn=refresh_names, inputs=[], outputs=[speaker_pick])


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # localhost only: browsers block mic capture over a LAN IP
        server_port=7861,         # 7861 so it can run next to app_ui.py on 7860
        theme=gr.themes.Soft(),
        css=BOOTH_CSS,
    )
