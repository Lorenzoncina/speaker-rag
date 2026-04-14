import os
import gradio as gr
import requests

# --- Configuration ---
FASTAPI_URL = "http://127.0.0.1:8006" #fbk server (requires SSH Port Forwarding)

# ==========================================
# Tab 1: Direct LLM Functions (No RAG)
# ==========================================
def ask_direct_llm(audio_paths, question):
    if not audio_paths:
        return "Please upload at least one audio file."
    
    if not question:
        return "Please provide a question."
        
    # Route the raw audio directly to the local FastAPI endpoint running Qwen-Audio
    files = [('audios', (os.path.basename(path), open(path, 'rb'), 'audio/wav')) for path in audio_paths]
    data = {'question': question}
    
    try:
        response = requests.post(f"{FASTAPI_URL}/llm/direct", files=files, data=data)
        if response.status_code == 200:
            res_json = response.json()
            return res_json.get('answer', str(res_json))
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is the FastAPI backend running on {FASTAPI_URL}?"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# ==========================================
# Tab 2: Speaker RAG Functions
# ==========================================
def enroll_rag(audio_paths, user_ids):
    if not audio_paths:
        return "Please upload audio files to enroll."
    
    # Format files for FastAPI's List[UploadFile]
    files = [('audios', (os.path.basename(path), open(path, 'rb'), 'audio/wav')) for path in audio_paths]
    
    # Only format user_ids if provided, allowing the backend to handle random UUIDs if absent
    data = {}
    if user_ids and user_ids.strip():
        data = {'user_ids': [uid.strip() for uid in user_ids.split(',')]}
        
    try:
        response = requests.post(f"{FASTAPI_URL}/rag/index", files=files, data=data)
        if response.status_code == 200:
            return f"Success:\n{response.json()}"
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is the FastAPI backend running on {FASTAPI_URL}?"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def list_enrolled_users():
    """Fetches and formats the list of all enrolled users."""
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
    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is the FastAPI backend running on {FASTAPI_URL}?"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def ask_rag(audio_paths, question):
    if not audio_paths:
        return "Please upload probe audio files.", ""
    
    files = [('audios', (os.path.basename(path), open(path, 'rb'), 'audio/wav')) for path in audio_paths]
    data = {'question': question}
    
    try:
        response = requests.post(f"{FASTAPI_URL}/rag/ask", files=files, data=data)
        if response.status_code == 200:
            res_json = response.json()
            
            # The backend now returns both the LLM's answer and the injected context
            answer = res_json.get('answer', "No answer received from the LLM.")
            context = res_json.get('context', "No context provided by the backend.")
            
            return answer, context
            
        else:
            return f"Error {response.status_code}:\n{response.text}", "Error fetching context."
            
    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is the FastAPI backend running on {FASTAPI_URL}?", ""
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

def clear_rag():
    try:
        response = requests.delete(f"{FASTAPI_URL}/rag/clear")
        if response.status_code == 200:
            return "Database cleared successfully."
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is the FastAPI backend running on {FASTAPI_URL}?"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def ingest_meld_ui():
    """Triggers the backend backdoor to ingest the MELD dataset."""
    try:
        # Note: We use GET here because that's how we defined it in endpoints.py
        response = requests.get(f"{FASTAPI_URL}/admin/ingest_meld")
        if response.status_code == 200:
            res_json = response.json()
            return f"MELD Ingestion Complete!\nProcessed: {res_json.get('enrolled_count', 0)} files."
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is the FastAPI backend running on {FASTAPI_URL}?"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# ==========================================
# Gradio UI Layout
# ==========================================
with gr.Blocks(title="Speaker Identification & Resolution Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ Speaker Identification & Resolution Demo")
    gr.Markdown("Compare the zero-shot capabilities of a Multimodal LLM against a custom Speaker RAG pipeline. **(Running 100% Locally on Qwen-Audio)**")
    
    with gr.Tabs():
        
        # --- TAB 1: Direct LLM ---
        with gr.Tab("1. Direct Multimodal LLM (No RAG)"):
            gr.Markdown("Upload audio files directly to the local Qwen-Audio model to test its raw identification and resolution capabilities without external vector databases.")
            
            with gr.Row():
                with gr.Column():
                    direct_audio = gr.File(label="Upload Audio Files", file_count="multiple", type="filepath")
                    direct_question = gr.Textbox(
                        label="Question", 
                        placeholder="e.g., How many speakers are there? Can you describe their gender and age?"
                    )
                    direct_btn = gr.Button("Ask Direct LLM", variant="primary")
                
                with gr.Column():
                    direct_output = gr.Textbox(label="LLM Response", lines=15)
            
            direct_btn.click(
                fn=ask_direct_llm, 
                inputs=[direct_audio, direct_question], 
                outputs=direct_output
            )
            
        # --- TAB 2: Speaker RAG Pipeline ---
        with gr.Tab("2. Speaker RAG Pipeline"):
            gr.Markdown("Use your custom backend: extracts attributes (Age, Gender, Language, Titanet Embeddings), searches ChromaDB, and passes context + audio to Qwen-Audio.")
            
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
                            meld_btn = gr.Button(" Ingest MELD (Admin)", variant="primary")
                    
                    with gr.Column():
                        enroll_output = gr.Textbox(label="Enrollment / Database Status", lines=10)
                
                enroll_btn.click(fn=enroll_rag, inputs=[enroll_audio, enroll_ids], outputs=enroll_output)
                list_btn.click(fn=list_enrolled_users, inputs=[], outputs=enroll_output)
                clear_btn.click(fn=clear_rag, inputs=[], outputs=enroll_output)
                meld_btn.click(fn=ingest_meld_ui, inputs=[], outputs=enroll_output)
                
            with gr.Accordion("🔍 Step 2: Query / Probe Audio", open=True):
                gr.Markdown("Upload unknown probe files to identify them against the enrolled database and ask the multimodal LLM questions about the results.")
                with gr.Row():
                    with gr.Column():
                        rag_audio = gr.File(label="Upload Probe Audio Files", file_count="multiple", type="filepath")
                        rag_question = gr.Textbox(
                            label="Question", 
                            placeholder="e.g., Who is speaking in this audio file? What are their attributes based on the backend results?"
                        )
                        rag_btn = gr.Button("Ask RAG Pipeline", variant="primary")
                    
                    with gr.Column():
                        rag_output = gr.Textbox(label="LLM Response", lines=6)
                        with gr.Accordion("⚙️ View Backend Context (Debug)", open=False):
                            rag_context = gr.Textbox(label="RAG Context Passed to LLM", lines=15)
                        
                rag_btn.click(fn=ask_rag, inputs=[rag_audio, rag_question], outputs=[rag_output, rag_context])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)