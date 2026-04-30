import os
import sys
import json
from pathlib import Path
import gradio as gr
import requests
import torch
import numpy as np
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

# ==========================================
# 🧠 "Bring Your Own LLM" Initialization
# ==========================================
print("Loading Local Text LLM for the UI...")
# You can swap this out for ANY text model (Llama-3, Mistral, etc.) or an OpenAI API call!
llm_pipeline = pipeline(
    "text-generation", 
    model="Qwen/Qwen2-7B-Instruct", # OR "meta-llama/Meta-Llama-3-8B-Instruct"
    device_map="auto",
    torch_dtype=torch.float16
)
print("UI LLM Loaded successfully.")

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

            messages = [
                {"role": "system", "content": "You are a helpful and precise analytical assistant."},
                {"role": "user", "content": recommended_prompt}
            ]

            formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            outputs = llm_pipeline(formatted_prompt, max_new_tokens=300, temperature=0.1)
            final_answer = outputs[0]["generated_text"][len(formatted_prompt):].strip()

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
    try:
        response = requests.get(
            f"{FASTAPI_URL}/admin/ingest_voxceleb",
            params={"per_speaker": int(per_speaker)},
            timeout=3600,
        )
        if response.status_code == 200:
            r = response.json()
            return (
                f"VoxCeleb1 Ingestion Complete!\n"
                f"Speakers: {r.get('n_speakers', 0)}\n"
                f"Per speaker: {r.get('per_speaker', 0)}\n"
                f"Total enrolled: {r.get('enrolled_count', 0)} files."
            )
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

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
            pairs, llm_pipeline, FASTAPI_URL, progress_callback=cb
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
    n = int(n_pairs) if int(n_pairs) > 0 else 200
    if regenerate or not Path(DEFAULT_VOX1_PAIRS_PATH).exists():
        build_vox1_pairs(n_pairs=n, output_path=DEFAULT_VOX1_PAIRS_PATH)
    return load_vox1_pairs(DEFAULT_VOX1_PAIRS_PATH)


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
            pairs, llm_pipeline, FASTAPI_URL, progress_callback=cb
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
# Gradio UI Layout
# ==========================================
with gr.Blocks(title="Speaker RAG Engine Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ Speaker RAG Engine Demo")
    gr.Markdown("This UI demonstrates a decoupled architecture. The backend extracts audio context and builds a Fact Sheet, which is then passed to a local Text LLM (`Qwen2-1.5B`) running in this UI script.")
    
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
            "Prior Q/A turns are stored client-side and sent along with each new question so the LLM can resolve follow-up references. Use **Reset Memory** to start a fresh conversation."
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
                    reset_btn = gr.Button("🧹 Reset Memory", variant="stop")

            with gr.Column():
                rag_output = gr.Textbox(label="LLM Response", lines=6)
                history_display = gr.Textbox(label="Conversation Memory", value="(no prior turns)", lines=8, interactive=False)
                with gr.Accordion("⚙️ View Backend Context (Debug)", open=False):
                    rag_context = gr.Textbox(label="RAG Fact Sheet (Returned by API)", lines=15)

        rag_btn.click(
            fn=ask_rag,
            inputs=[rag_audio, rag_question, conversation_state],
            outputs=[rag_output, rag_context, conversation_state, history_display],
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
            "Use *Number of pairs* to change the subset size; tick *Regenerate pairs* "
            "to rebuild the JSON with a new subset."
        )
        with gr.Row():
            vox1_eval_n_pairs = gr.Number(
                label="Number of pairs (-1 = all 37,720)",
                value=200, precision=0,
            )
            vox1_eval_regen = gr.Checkbox(label="Regenerate pairs", value=False)

        with gr.Row():
            vox1_exp1_btn = gr.Button("Vox1 Experiment 1: Qwen2-Audio direct", variant="primary")
            vox1_exp2_btn = gr.Button("Vox1 Experiment 2: RAG (empty DB) + text LLM", variant="primary")
            vox1_exp3_btn = gr.Button("Vox1 Experiment 3: RAG (VoxCeleb1 enrolled) + text LLM", variant="primary")

        vox1_eval_output = gr.Textbox(label="VoxCeleb1 Evaluation Result", lines=10)

        vox1_exp1_btn.click(fn=run_vox1_exp1,
                            inputs=[vox1_eval_n_pairs, vox1_eval_regen],
                            outputs=vox1_eval_output)
        vox1_exp2_btn.click(fn=run_vox1_exp2,
                            inputs=[vox1_eval_n_pairs, vox1_eval_regen],
                            outputs=vox1_eval_output)
        vox1_exp3_btn.click(fn=run_vox1_exp3,
                            inputs=[vox1_eval_n_pairs, vox1_eval_regen],
                            outputs=vox1_eval_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)