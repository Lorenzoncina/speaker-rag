"""
Script Name: diarization_service.py

Description:
Handles the ingestion of full multi-speaker conversations (e.g. Teams calls, phone calls).
Given a raw audio file, this service:
  1. Runs Voice Activity Detection (VAD) via pyannote.audio to strip silence/noise.
  2. Segments speaker turns using pyannote's speaker diarization pipeline.
  3. Extracts Titanet speaker embeddings per segment.
  4. Clusters segments into speaker groups using agglomerative clustering.
  5. Matches cluster centroids against already-enrolled speakers for cross-call identity.
  6. Exports segmented audio files ready for the existing enrollment pipeline.

The user may optionally supply per-segment speaker labels (e.g. from Teams/Meet metadata).
When labels are provided, clustering is skipped and segments are assigned directly.

Author: Lorenzo Concina
"""
import os
import re
import uuid
import tempfile
import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering

SIMILARITY_THRESHOLD = 0.50


def parse_vtt(vtt_path: str) -> List[Dict[str, Any]]:
    """Parses a WebVTT file (e.g. from Microsoft Teams) into speaker turns.

    Handles the Teams VTT format:
        00:00:03.874 --> 00:00:04.554
        <v Speaker Name>Transcribed text.</v>

    Consecutive cues from the same speaker are merged into a single turn.

    Returns a list of dicts sorted by start time:
      [{"start": float, "end": float, "speaker_label": str, "transcription": str}, ...]
    """
    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    timestamp_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
    )
    speaker_re = re.compile(r"<v\s+([^>]+)>(.+?)</v>", re.DOTALL)

    raw_cues: List[Dict[str, Any]] = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        ts_match = timestamp_re.search(lines[i])
        if ts_match:
            g = ts_match.groups()
            start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000.0
            end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000.0

            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            full_text = " ".join(text_lines)
            spk_match = speaker_re.search(full_text)
            if spk_match:
                speaker = spk_match.group(1).strip()
                transcript = spk_match.group(2).strip()
            else:
                speaker = "Unknown"
                transcript = full_text.strip()

            raw_cues.append({
                "start": start,
                "end": end,
                "speaker_label": speaker,
                "transcription": transcript,
            })
        i += 1

    if not raw_cues:
        return []

    merged: List[Dict[str, Any]] = [raw_cues[0].copy()]
    for cue in raw_cues[1:]:
        prev = merged[-1]
        if cue["speaker_label"] == prev["speaker_label"]:
            prev["end"] = cue["end"]
            prev["transcription"] += " " + cue["transcription"]
        else:
            merged.append(cue.copy())

    return merged


class _Diarization_Service:
    _instance = None
    _pipeline = None
    _embedding_model = None
    _device = None

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Runs pyannote speaker diarization on a full conversation audio file.

        Returns a list of turn dicts sorted by start time:
          [{"start": float, "end": float, "speaker_label": str}, ...]
        The speaker_label is pyannote's internal cluster label (e.g. SPEAKER_00).
        """
        output = self._pipeline(audio_path)
        # Resolve annotation regardless of pyannote version:
        #   3.x → pipeline returns Annotation directly (has itertracks)
        #   4.x → pipeline returns DiarizeOutput dataclass; field is .diarization or .annotation
        if hasattr(output, "itertracks"):
            annotation = output                          # pyannote 3.x: bare Annotation
        elif hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization     # pyannote 4.x: DiarizeOutput
        elif hasattr(output, "diarization"):
            annotation = output.diarization
        elif hasattr(output, "annotation"):
            annotation = output.annotation
        else:
            raise ValueError(
                f"Unrecognised pyannote pipeline output type '{type(output).__name__}'. "
                f"Available attributes: {[a for a in dir(output) if not a.startswith('_')]}"
            )
        turns = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            turns.append({
                "start": segment.start,
                "end": segment.end,
                "speaker_label": label,
            })
        turns.sort(key=lambda t: t["start"])
        return turns

    def segment_audio(
        self,
        audio_path: str,
        turns: List[Dict[str, Any]],
        output_dir: str,
        min_duration: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Slices the full audio into per-turn WAV files.

        Skips segments shorter than min_duration seconds.
        Returns enriched turn dicts with an added "segment_path" key.
        """
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        segments = []
        for i, turn in enumerate(turns):
            duration = turn["end"] - turn["start"]
            if duration < min_duration:
                continue
            start_sample = int(turn["start"] * sr)
            end_sample = int(turn["end"] * sr)
            chunk = waveform[start_sample:end_sample]

            seg_path = os.path.join(output_dir, f"seg_{i:04d}_{turn['speaker_label']}.wav")
            sf.write(seg_path, chunk, sr)

            segments.append({
                **turn,
                "segment_path": seg_path,
                "duration": round(duration, 3),
                "utterance_id": i,
            })
        return segments

    def compute_segment_embeddings(
        self, segments: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Computes Titanet speaker embeddings for each segment.

        Returns (segments, embeddings_matrix) where embeddings_matrix has shape (N, 192)
        and segments are enriched with an "embedding" key.
        """
        all_embeddings = []
        for seg in segments:
            x, sr = librosa.load(seg["segment_path"], sr=16000, mono=True)
            sig = torch.from_numpy(x).unsqueeze(0).to(self._device)
            sig_len = torch.tensor([sig.shape[1]]).to(self._device)
            with torch.no_grad():
                _, embedding = self._embedding_model.forward(
                    input_signal=sig, input_signal_length=sig_len
                )
            emb_np = embedding.squeeze().cpu().numpy()
            seg["embedding"] = emb_np
            all_embeddings.append(emb_np)

        return segments, np.stack(all_embeddings) if all_embeddings else np.array([])

    def cluster_segments(
        self,
        segments: List[Dict[str, Any]],
        embeddings: np.ndarray,
        num_speakers: Optional[int] = None,
        distance_threshold: float = 0.5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Groups segments by speaker using agglomerative clustering on Titanet embeddings.

        If num_speakers is given, uses it directly. Otherwise uses distance_threshold to
        auto-detect the number of clusters.

        Returns a dict mapping cluster_label -> list of segments belonging to that cluster.
        Each segment gets an added "cluster_label" key.
        """
        if len(segments) == 0:
            return {}
        if len(segments) == 1:
            segments[0]["cluster_label"] = "speaker_0"
            return {"speaker_0": segments}

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = embeddings / norms

        if num_speakers is not None:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                metric="cosine",
                linkage="average",
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )

        labels = clustering.fit_predict(normed)

        clusters: Dict[str, List[Dict[str, Any]]] = {}
        for seg, label in zip(segments, labels):
            cluster_label = f"speaker_{label}"
            seg["cluster_label"] = cluster_label
            clusters.setdefault(cluster_label, []).append(seg)

        return clusters

    def match_clusters_to_enrolled(
        self,
        clusters: Dict[str, List[Dict[str, Any]]],
        identification_service,
    ) -> Dict[str, str]:
        """Computes a centroid per cluster and matches it against enrolled speakers.

        Returns a mapping cluster_label -> resolved identity (enrolled user_id or a
        generated unknown label).
        """
        identity_map: Dict[str, str] = {}

        for cluster_label, segs in clusters.items():
            centroid = np.mean([s["embedding"] for s in segs], axis=0)
            centroid_list = centroid.tolist()

            try:
                tmp_id = f"_tmp_centroid_{uuid.uuid4()}"
                identification_service._collection.add(
                    ids=[tmp_id],
                    embeddings=[centroid_list],
                    metadatas=[{"user_id": tmp_id, "data_type": "probe", "session_id": "_cluster_match"}],
                )
                results = identification_service._collection.query(
                    query_embeddings=[centroid_list],
                    n_results=5,
                    include=["metadatas", "distances"],
                    where={"data_type": "enroll"},
                )
                identification_service._collection.delete(ids=[tmp_id])

                best_score = 0.0
                best_uid = None
                if results and results.get("ids") and results["ids"][0]:
                    for res_id, meta, dist in zip(
                        results["ids"][0], results["metadatas"][0], results["distances"][0]
                    ):
                        score = max(0.0, 1.0 - dist)
                        if score > best_score:
                            best_score = score
                            best_uid = meta.get("user_id")

                if best_uid and best_score >= SIMILARITY_THRESHOLD:
                    identity_map[cluster_label] = best_uid
                else:
                    identity_map[cluster_label] = f"unknown_{cluster_label}"

            except Exception as e:
                print(f"Cluster matching failed for {cluster_label}: {e}")
                identity_map[cluster_label] = f"unknown_{cluster_label}"

        return identity_map

    def process_conversation(
        self,
        audio_path: str,
        speaker_map: Optional[Dict[str, str]] = None,
        num_speakers: Optional[int] = None,
        identification_service=None,
        vtt_turns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: diarize -> segment -> embed -> cluster -> match identities.

        Parameters:
        - audio_path: path to the full conversation audio file.
        - speaker_map: optional dict mapping pyannote labels to known names
                       (e.g. {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}).
                       When provided, clustering is skipped and labels are applied directly.
        - num_speakers: optional hint for the number of speakers (helps clustering).
        - identification_service: the active Identification_Service singleton for cross-call
                                  identity matching. If None, no cross-call matching is done.
        - vtt_turns: optional list of pre-parsed VTT turns from parse_vtt().
                     When provided, pyannote diarization is skipped entirely. Each turn
                     already has speaker_label (the real name) and an optional transcription.

        Returns a dict with:
          - dialogue_id: unique ID for this conversation
          - segments: list of enriched segment dicts (with identity, cluster, paths, etc.)
          - identity_map: cluster_label -> resolved identity
          - clusters: cluster_label -> [segment, ...]
        """
        dialogue_id = str(uuid.uuid4())
        print(f"[Diarization] Processing conversation: {os.path.basename(audio_path)}")

        output_dir = tempfile.mkdtemp(prefix="diarize_segments_")

        if vtt_turns:
            print(f"[Diarization] Using VTT transcript ({len(vtt_turns)} turns, pyannote skipped).")
            turns = vtt_turns
        else:
            print(f"[Diarization] Running speaker diarization...")
            turns = self.diarize(audio_path)
            print(f"[Diarization] Found {len(turns)} raw turns.")

        segments = self.segment_audio(audio_path, turns, output_dir)
        print(f"[Diarization] Extracted {len(segments)} valid segments (>0.5s).")

        if not segments:
            return {
                "dialogue_id": dialogue_id,
                "segments": [],
                "identity_map": {},
                "clusters": {},
            }

        # Carry over VTT transcriptions to segments
        if vtt_turns:
            vtt_by_start = {round(t["start"], 3): t for t in vtt_turns}
            for seg in segments:
                vtt_turn = vtt_by_start.get(round(seg["start"], 3))
                if vtt_turn and vtt_turn.get("transcription"):
                    seg["vtt_transcription"] = vtt_turn["transcription"]

        print(f"[Diarization] Computing Titanet embeddings for segments...")
        segments, embeddings = self.compute_segment_embeddings(segments)

        if vtt_turns:
            # VTT already provides real speaker names — use them directly
            clusters: Dict[str, List[Dict[str, Any]]] = {}
            for seg in segments:
                name = seg["speaker_label"]
                seg["cluster_label"] = name
                clusters.setdefault(name, []).append(seg)
            identity_map = {name: name for name in clusters}
            print(f"[Diarization] Applied VTT speaker names: {list(clusters.keys())}")
        elif speaker_map:
            clusters = {}
            for seg in segments:
                mapped_name = speaker_map.get(seg["speaker_label"], seg["speaker_label"])
                seg["cluster_label"] = mapped_name
                clusters.setdefault(mapped_name, []).append(seg)
            identity_map = {name: name for name in clusters}
            print(f"[Diarization] Applied provided speaker map: {list(speaker_map.values())}")
        else:
            print(f"[Diarization] Clustering segments (num_speakers={num_speakers})...")
            clusters = self.cluster_segments(
                segments, embeddings, num_speakers=num_speakers
            )
            print(f"[Diarization] Found {len(clusters)} speaker clusters.")

            if identification_service:
                print(f"[Diarization] Matching clusters against enrolled speakers...")
                identity_map = self.match_clusters_to_enrolled(clusters, identification_service)
            else:
                identity_map = {cl: f"unknown_{cl}" for cl in clusters}

        for seg in segments:
            seg["dialogue_id"] = dialogue_id
            seg["identity"] = identity_map.get(seg["cluster_label"], seg["cluster_label"])

        return {
            "dialogue_id": dialogue_id,
            "segments": segments,
            "identity_map": identity_map,
            "clusters": clusters,
        }


def Diarization_Service(embedding_model, device: str, hf_auth_token: str = None):
    """Factory function for the singleton diarization service.

    Parameters:
    - embedding_model: the pre-loaded Titanet model (shared with Identification_Service).
    - device: "cuda" or "cpu".
    - hf_auth_token: HuggingFace auth token required by pyannote (gated model).
    """
    if _Diarization_Service._instance is None:
        # Set embedding model and device first (no failure risk)
        _Diarization_Service._embedding_model = embedding_model
        _Diarization_Service._device = device

        print("[Diarization] Loading pyannote speaker diarization pipeline...")
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio is required for conversation diarization. "
                "Install it with: pip install pyannote.audio==4.0.4"
            )
        pipeline_kwargs = {}
        if hf_auth_token:
            # pyannote 4.x uses 'token'; 'use_auth_token' was removed in HF Hub >=0.17
            pipeline_kwargs["token"] = hf_auth_token
        _Diarization_Service._pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", **pipeline_kwargs
        )
        if device == "cuda" and torch.cuda.is_available():
            _Diarization_Service._pipeline.to(torch.device("cuda"))

        print("[Diarization] Pipeline loaded successfully.")
        # Assign _instance only after everything succeeds so a failed attempt
        # does not poison the singleton and block future retries.
        _Diarization_Service._instance = _Diarization_Service()

    return _Diarization_Service._instance
