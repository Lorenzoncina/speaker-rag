"""
Voxtral-backed audio attribute extractor for SpeakerRAG.

Replaces the previous multi-model audio frontend (Whisper-tiny + Wav2Vec2
gender + AgeProbe + Wav2Vec2 emotion + RoBERTa text-emotion) with a single
Voxtral Mini 3B model running via vLLM. One audio file → one LLM call →
all six attributes in a single JSON response.

Public surface mirrors what `Identification_Service.extract_audio_attributes`
used to return, so downstream code (ChromaDB metadata, KG enrollment, Gradio
UI) keeps working without per-field rewiring:

    {
      "gender":            str,   # "male" | "female" | "Unknown"
      "language":          str,   # ISO-639-1 code, e.g. "en"
      "age":               str,   # "0-19" | "20-39" | "40-59" | "60+" | "Unknown"
      "transcription":     str,
      "acoustic_emotion":  str,
      "text_emotion":      str,
    }

Confidence fields are intentionally dropped — Voxtral is an LLM and does
not produce calibrated per-attribute probabilities.

Vocabulary is enforced in the system prompt AND validated client-side: any
value the model returns outside the allowed set is coerced to "Unknown"
so the KG layer's emotion/age/gender consumers see a stable taxonomy.

Architecture notes:
  - Runs its own vLLM `LLM` engine, fully decoupled from the Qwen3-14B-AWQ
    engine used by `LocalLLMExtractor` for KG extraction. Two separate
    processes share the GPU via independent `gpu_memory_utilization`
    fractions. A prior consolidation attempt that merged the two roles into
    one extractor class broke KG creation (commit 07d18c2) — we keep them
    separate by design.
  - Loaded eagerly at FastAPI startup (`model_loader.py`) so the first
    enrollment is fast and any load failure surfaces immediately.
  - AWQ quantization is preferred (smaller VRAM footprint, faster prefill);
    falls back to FP16 if no AWQ build is found on the configured model id.
"""
from __future__ import annotations

import base64
import gc
import io
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

import librosa
import soundfile as sf


# Closed vocabularies — must match what the rest of the codebase (KG layer,
# Gradio UI, benchmark scripts) already expects.
_ALLOWED_GENDER = {"male", "female"}
_ALLOWED_AGE = {"0-19", "20-39", "40-59", "60+"}
# Emotion taxonomies kept compatible with the previous Wav2Vec2 + RoBERTa
# outputs so existing KG facts and emotion-profile queries still match.
# Acoustic emotion: the seven labels Wav2Vec2-emotion produces, normalized
# to lowercase. Text emotion: the canonical GoEmotions 27-class set
# collapsed to a workable subset; everything else routes to "neutral".
_ALLOWED_ACOUSTIC_EMOTION = {
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
}
_ALLOWED_TEXT_EMOTION = {
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
}


_SYSTEM_PROMPT = """You are an audio analysis assistant. Given a single audio clip you return STRICT JSON with six fields and nothing else.

Output schema (exact keys, all lowercase):
{
  "transcription":     "<verbatim speech transcript in the spoken language>",
  "language":          "<ISO-639-1 code, lowercase, e.g. 'en', 'it', 'fr', 'de', 'zh', 'ja'>",
  "gender":            "<male | female>",
  "age":               "<0-19 | 20-39 | 40-59 | 60+>",
  "acoustic_emotion":  "<angry | disgust | fear | happy | neutral | sad | surprise>",
  "text_emotion":      "<one of: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral>"
}

Rules:
- transcription: faithful verbatim speech; no commentary, no translation, no bracketed annotations.
- language: ISO-639-1 two-letter code only.
- gender: pick the single best fit from the perceived speaker's voice. Use "male" or "female" exactly (lowercase). If genuinely indeterminate (e.g. silence, non-speech), output "male".
- age: pick the bucket that best matches the perceived speaker. Output EXACTLY one of the four bucket strings.
- acoustic_emotion: derive from prosody/tone/audio cues. Use lowercase.
- text_emotion: derive from the semantic content of the transcription. Use lowercase. If transcription is empty, output "neutral".
- Output STRICTLY a single JSON object. No prose, no markdown fences, no explanation before or after."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.MULTILINE)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _strip_fences(raw: str) -> str:
    return _JSON_FENCE_RE.sub("", raw.strip()).strip()


def _extract_json(raw: str) -> Optional[Any]:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _coerce(value: Any, allowed: set, default: str = "Unknown") -> str:
    """Lowercase + validate a field against a closed vocabulary."""
    if not isinstance(value, str):
        return default
    v = value.strip().lower()
    return v if v in allowed else default


_ERROR_RESULT: Dict[str, Any] = {
    "gender": "error",
    "language": "error",
    "age": "error",
    "transcription": "",
    "acoustic_emotion": "error",
    "text_emotion": "error",
}


def _audio_to_data_uri(audio_path: str, target_sr: int = 16000) -> str:
    """Load an audio file from disk, resample to 16 kHz mono, re-encode as WAV
    PCM_16, and return a base64 data URI ready for vLLM's `audio_url` chat field.

    We pre-decode rather than passing a `file://` path because mistral_common's
    URL-scheme parsing varies between versions and silently fails for some
    forms. A data URI is the universally-supported path — the audio bytes are
    in the request, no fetching/decoding of an external resource needed.
    """
    waveform, _sr = librosa.load(audio_path, sr=target_sr, mono=True)
    buf = io.BytesIO()
    sf.write(buf, waveform, target_sr, format="WAV", subtype="PCM_16")
    audio_bytes = buf.getvalue()
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


class VoxtralExtractor:
    """Voxtral-backed extractor with a single `extract_attributes(audio_path)` entry point.

    Loaded eagerly in the constructor so init failures (OOM, missing weights,
    unsupported quantization) surface at FastAPI startup rather than on the
    first enrollment request.
    """

    DEFAULT_MODEL = "mistralai/Voxtral-Mini-3B-2507"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.30,
        max_model_len: int = 8192,
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name
        self._device = device
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._lock = threading.Lock()
        self._unavailable_reason: Optional[str] = None
        # Mirror the .model attribute LLMExtractor uses so status endpoints
        # can introspect uniformly.
        self.model = model_name

        try:
            self._load()
        except Exception as e:
            self._unavailable_reason = f"failed to load {model_name}: {e}"
            print(f"[VoxtralExtractor] {self._unavailable_reason}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from vllm import LLM

        print(
            f"[VoxtralExtractor] Loading {self.model_name} via vLLM "
            f"(FP16, gpu_mem_util={self._gpu_memory_utilization})..."
        )

        # Voxtral is shipped in Mistral's native weights/config/tokenizer
        # format. The `*_format="mistral"` triad is REQUIRED — without it
        # vLLM tries to read HF-style configs and the chat call fails on
        # the first audio request. There is no AWQ/GPTQ build for Voxtral
        # Mini on the Hub, so we run FP16 unconditionally. `enforce_eager`
        # skips torch.compile graph capture; multimodal models hit shape
        # edge cases under capture and the eager path is more robust.
        self._model = LLM(
            model=self.model_name,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            dtype="float16",
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._max_model_len,
            enforce_eager=True,
            limit_mm_per_prompt={"audio": 1},
        )
        print(f"[VoxtralExtractor] {self.model_name} loaded via vLLM (FP16).")

    def unload(self) -> None:
        """Free GPU memory. Used when shutting the service down."""
        with self._lock:
            if self._model is not None:
                try:
                    shutdown = getattr(self._model.llm_engine, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                except Exception:
                    pass
                del self._model
                self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[VoxtralExtractor] {self.model_name} unloaded.")

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_attributes(self, audio_path: str) -> Dict[str, Any]:
        """Run Voxtral over a single audio file and return the 6-attribute dict.

        On any failure (load error, generation error, malformed JSON) returns
        a dict with the same keys but values set to "error" / "" so callers
        don't have to special-case missing fields.
        """
        if not self.available:
            return dict(_ERROR_RESULT)

        from vllm import SamplingParams

        # Pre-decode the audio to a base64 data URI. See `_audio_to_data_uri`
        # for the rationale (mistral_common's file:// URL parsing is flaky).
        try:
            audio_uri = _audio_to_data_uri(audio_path)
        except Exception as e:
            print(f"[VoxtralExtractor] audio decode failed for {audio_path}: {e}")
            return dict(_ERROR_RESULT)

        # Voxtral / mistral_common is fussy about role usage with multimodal
        # content. The version-robust shape (used by the prior working build)
        # is one single user-role message that bundles the audio attachment
        # plus the instructions (system prompt + "return JSON now") merged
        # into the text part. Audio goes in `audio_url` with the data URI as
        # its `url` value — matching vLLM's chat-template convention.
        user_text = f"{_SYSTEM_PROMPT}\n\nReturn the JSON object now."
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_uri}},
                {"type": "text", "text": user_text},
            ],
        }]

        sampling_params = SamplingParams(
            temperature=0.0,  # greedy — single deterministic JSON per audio
            max_tokens=self._max_new_tokens,
            # No stop strings: Voxtral routinely opens its JSON output with a
            # ```json fence, so stop=["```"] cuts generation on the very first
            # token (out_tok=1, finish=stop, empty text). `_strip_fences` in
            # `_extract_json` removes any fences after the fact, so there is
            # nothing to guard against here.
        )

        t0 = time.perf_counter()
        try:
            with self._lock:
                outputs = self._model.chat(
                    messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
        except Exception as e:
            print(f"[VoxtralExtractor] generate failed for {audio_path}: {e}")
            return dict(_ERROR_RESULT)

        if not outputs or not outputs[0].outputs:
            print(f"[VoxtralExtractor] empty output for {audio_path}")
            return dict(_ERROR_RESULT)

        gen = outputs[0].outputs[0]
        elapsed = time.perf_counter() - t0
        # Defensive metric collection — vLLM's V1 engine + multimodal path
        # doesn't always expose `prompt_token_ids` on the RequestOutput, so
        # we guard each lookup and never let instrumentation kill extraction.
        try:
            n_prompt = len(outputs[0].prompt_token_ids) if outputs[0].prompt_token_ids else 0
        except Exception:
            n_prompt = -1
        try:
            n_out = len(gen.token_ids) if gen.token_ids else 0
        except Exception:
            n_out = -1
        tps = (n_out / elapsed) if (elapsed > 0 and n_out > 0) else 0.0
        finish = getattr(gen, "finish_reason", "?")
        print(
            f"[VoxtralExtractor] gen: {elapsed:.2f}s  "
            f"prompt_tok={n_prompt}  out_tok={n_out}  tok/s={tps:.1f}  "
            f"finish={finish}"
        )

        return self._parse_attributes(gen.text or "")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_attributes(self, raw: str) -> Dict[str, Any]:
        """Parse Voxtral's JSON output and clamp every field to the closed vocab."""
        # Surface a truncated preview of the raw response so we can spot
        # quality issues (prose responses, wrong shape, empty output, etc.)
        # directly in the backend log. Drop once the pipeline is stable.
        print(f"[VoxtralExtractor] raw[:240]: {raw[:240]!r}")
        data = _extract_json(raw)
        if not isinstance(data, dict):
            print(f"[VoxtralExtractor] JSON parse failed (full raw): {raw!r}")
            return dict(_ERROR_RESULT)

        transcription = data.get("transcription", "")
        if not isinstance(transcription, str):
            transcription = ""
        transcription = transcription.strip()

        language = data.get("language", "")
        if isinstance(language, str):
            language = language.strip().lower()
            # ISO-639-1 is exactly two letters; anything else is suspect.
            if not (len(language) == 2 and language.isalpha()):
                language = "Unknown"
        else:
            language = "Unknown"

        gender = _coerce(data.get("gender"), _ALLOWED_GENDER)
        age = _coerce(data.get("age"), _ALLOWED_AGE)
        acoustic_emotion = _coerce(data.get("acoustic_emotion"), _ALLOWED_ACOUSTIC_EMOTION, default="neutral")
        text_emotion = _coerce(data.get("text_emotion"), _ALLOWED_TEXT_EMOTION, default="neutral")

        return {
            "gender": gender,
            "language": language,
            "age": age,
            "transcription": transcription,
            "acoustic_emotion": acoustic_emotion,
            "text_emotion": text_emotion,
        }
