import os
import gradio as gr
import requests

# --- Configuration ---
# Point this to where your endpoints.py FastAPI server is running
FASTAPI_URL = "http://0.0.0.0:8006"

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

def ask_rag(audio_paths, question):
    if not audio_paths:
        return "Please upload probe audio files."
    
    files = [('audios', (os.path.basename(path), open(path, 'rb'), 'audio/wav')) for path in audio_paths]
    data = {'question': question}
    
    try:
        response = requests.post(f"{FASTAPI_URL}/rag/ask", files=files, data=data)
        if response.status_code == 200:
            res_json = response.json()
            # The backend returns a dictionary containing 'answer', 'question', and 'timestamp'
            return res_json.get('answer', str(res_json))
        else:
            return f"Error {response.status_code}:\n{response.text}"
    except requests.exceptions.ConnectionError:
        return f"Connection Error: Is the FastAPI backend running on {FASTAPI_URL}?"
    except Exception as e:
        return f"An error occurred: {str(e)}"

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
                            clear_btn = gr.Button("Clear Entire Database", variant="stop")
                    
                    with gr.Column():
                        enroll_output = gr.Textbox(label="Enrollment / Database Status", lines=6)
                
                enroll_btn.click(fn=enroll_rag, inputs=[enroll_audio, enroll_ids], outputs=enroll_output)
                clear_btn.click(fn=clear_rag, inputs=[], outputs=enroll_output)
                
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
                        rag_output = gr.Textbox(label="RAG Pipeline Response", lines=12)
                        
                rag_btn.click(fn=ask_rag, inputs=[rag_audio, rag_question], outputs=rag_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)