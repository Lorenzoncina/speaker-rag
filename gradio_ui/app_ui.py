import os
import gradio as gr
import requests
import torch
from transformers import pipeline

# --- Configuration ---
FASTAPI_URL = "http://127.0.0.1:8006" # fbk server (requires SSH Port Forwarding)

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

def ask_rag(audio_paths, question):
    if not question or not question.strip():
        return "Please provide a question.", ""

    # Hybrid: audio optional. If missing, the backend extracts speaker names from the question.
    files = None
    if audio_paths:
        files = [('audios', (os.path.basename(path), open(path, 'rb'), 'audio/wav')) for path in audio_paths]

    data = {'question': question}

    try:
        # ---> UPDATED ENDPOINT <---
        response = requests.post(f"{FASTAPI_URL}/rag/retrieve", files=files, data=data)
        if response.status_code == 200:
            res_json = response.json()
            
            fact_sheet = res_json.get('fact_sheet', "No context provided by the backend.")
            recommended_prompt = res_json.get('recommended_prompt', "")
            
            if not recommended_prompt:
                return "Error: Backend did not return a prompt.", fact_sheet
                
            # 2. Pass the Context to OUR local UI LLM
            messages = [
                {"role": "system", "content": "You are a helpful and precise analytical assistant."},
                {"role": "user", "content": recommended_prompt}
            ]
            
            formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            outputs = llm_pipeline(formatted_prompt, max_new_tokens=300, temperature=0.1)
            final_answer = outputs[0]["generated_text"][len(formatted_prompt):]
            
            return final_answer.strip(), fact_sheet
            
        else:
            return f"Error {response.status_code}:\n{response.text}", "Error fetching context."
            
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

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
            
            with gr.Column():
                enroll_output = gr.Textbox(label="Enrollment / Database Status", lines=10)
        
        enroll_btn.click(fn=enroll_rag, inputs=[enroll_audio, enroll_ids], outputs=enroll_output)
        list_btn.click(fn=list_enrolled_users, inputs=[], outputs=enroll_output)
        clear_btn.click(fn=clear_rag, inputs=[], outputs=enroll_output)
        meld_btn.click(fn=ingest_meld_ui, inputs=[], outputs=enroll_output)
        
    with gr.Accordion("🔍 Step 2: Query / Probe Audio", open=True):
        gr.Markdown(
            "Ask a question with or without audio:\n"
            "- **With audio**: the backend identifies the speaker(s) and filters context by identity.\n"
            "- **Text only**: the backend scans the question for enrolled speaker names; if any match, it filters by them, otherwise it runs an unfiltered semantic search."
        )
        with gr.Row():
            with gr.Column():
                rag_audio = gr.File(label="Upload Probe Audio Files (optional)", file_count="multiple", type="filepath")
                rag_question = gr.Textbox(
                    label="Question",
                    placeholder="e.g., Are the speakers the same? What did Alice say about the project?"
                )
                rag_btn = gr.Button("Extract Context & Ask LLM", variant="primary")

            with gr.Column():
                rag_output = gr.Textbox(label="LLM Response", lines=6)
                with gr.Accordion("⚙️ View Backend Context (Debug)", open=False):
                    rag_context = gr.Textbox(label="RAG Fact Sheet (Returned by API)", lines=15)

        rag_btn.click(fn=ask_rag, inputs=[rag_audio, rag_question], outputs=[rag_output, rag_context])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)