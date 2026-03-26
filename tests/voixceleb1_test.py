import os
import glob
import requests

# Configuration
BASE_URL = "http://localhost:8006"
TEST_WAV_DIR = "/home/lorenzo-concina/development/datasets/VoxCeleb1/wav/test"
META_CSV_PATH = "/home/lorenzo-concina/development/datasets/VoxCeleb1/vox1_meta.csv"  # Ensure this path is correct
MAX_SPEAKERS = 5      
ENROLL_PER_SPK = 3     
PROBES_PER_SPK = 2     


def clear_backend_database():
    """Calls the API to wipe the ChromaDB collection before running tests."""
    print("--- Wiping existing vector database ---")
    try:
        response = requests.delete(f"{BASE_URL}/rag/clear")
        if response.status_code == 200:
            print("Database cleared successfully!")
        else:
            print(f"Warning: Failed to clear database: {response.text}")
    except Exception as e:
        print(f"Error connecting to server to clear database: {e}")


def load_speaker_names(csv_path):
    """Parses the VoxCeleb metadata to map IDs to actual names."""
    id_to_name = {}
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            # Skip the header line
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    spk_id = parts[0]
                    # Convert 'Eddie_Griffin' to 'Eddie Griffin'
                    clean_name = parts[1].replace("_", " ")
                    id_to_name[spk_id] = clean_name
    except FileNotFoundError:
        print(f"Warning: Could not find {csv_path}. Make sure the path is correct.")
    return id_to_name

def get_speaker_files(base_dir, max_speakers):
    """Gathers wav files grouped by speaker ID."""
    speaker_files = {}
    if not os.path.exists(base_dir):
        return speaker_files
        
    speaker_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for spk_id in speaker_dirs[:max_speakers]:
        search_pattern = os.path.join(base_dir, spk_id, "**", "*.wav")
        wavs = glob.glob(search_pattern, recursive=True)
        
        if len(wavs) >= (ENROLL_PER_SPK + PROBES_PER_SPK):
            speaker_files[spk_id] = wavs
            
    return speaker_files

def enroll_speakers(speaker_files, id_to_name):
    """Indexes the enrollment files using the actual speaker names."""
    print("\n--- Starting Enrollment Phase ---")
    for spk_id, wavs in speaker_files.items():
        # Fallback to the ID if the name isn't in the CSV for some reason
        speaker_name = id_to_name.get(spk_id, spk_id) 
        enroll_wavs = wavs[:ENROLL_PER_SPK]
        
        print(f"Enrolling '{speaker_name}' (ID: {spk_id}) with {len(enroll_wavs)} files...")
        
        files_to_upload = []
        file_handles = []
        
        for w in enroll_wavs:
            f = open(w, 'rb')
            file_handles.append(f)
            files_to_upload.append(('audios', (os.path.basename(w), f, 'audio/wav')))
        
        try:
            # We now pass the actual name to the API instead of the ID
            response = requests.post(
                f"{BASE_URL}/rag/index", 
                files=files_to_upload, 
                data={'user_ids': speaker_name}
            )
            if response.status_code != 200:
                print(f"Error enrolling {speaker_name}: {response.text}")
        finally:
            for f in file_handles:
                f.close()

def test_accuracy(speaker_files, id_to_name):
    """Queries the RAG system and checks if it returns the correct name."""
    print("\n--- Starting Testing Phase ---")
    correct = 0
    total = 0
    
    # Updated prompt to explicitly ask for the Name
    question = """Analyze the backend results and identify the speaker.
                You must output ONLY the name. Do not explain your reasoning.

                Example 1:
                Results: Match #1: Speaker ID 'Brad Pitt' (Similarity: 0.95)...
                Answer: Brad Pitt

                Now do the following:
                Who is the speaker in the provided backend results?
                Answer:"""

    for spk_id, wavs in speaker_files.items():
        true_speaker_name = id_to_name.get(spk_id, spk_id)
        probe_wavs = wavs[ENROLL_PER_SPK : ENROLL_PER_SPK + PROBES_PER_SPK]
        
        for probe in probe_wavs:
            print(f"Testing probe for '{true_speaker_name}': {os.path.basename(probe)}...")
            
            with open(probe, 'rb') as f:
                files_to_upload = [('audios', (os.path.basename(probe), f, 'audio/wav'))]
                data = {'question': question}
                
                response = requests.post(
                    f"{BASE_URL}/rag/ask", 
                    files=files_to_upload, 
                    data=data
                )
                
                if response.status_code == 200:
                    answer = response.json().get('answer', '')
                    
                    # We check if the true name is in the answer, ignoring case differences
                    is_match = true_speaker_name.lower() in answer.lower()
                    if is_match:
                        correct += 1
                    
                    print(f"  -> Predicted: {answer.strip()} | Match: {is_match}")
                else:
                    print(f"  -> Error: {response.status_code}")
                    
                total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n=== Final Accuracy: {correct}/{total} ({accuracy:.2f}%) ===")

if __name__ == "__main__":
    name_mapping = load_speaker_names(META_CSV_PATH)
    files_map = get_speaker_files(TEST_WAV_DIR, MAX_SPEAKERS)
    
    clear_backend_database()

    if not files_map:
        print("No audio files found. Check your TEST_WAV_DIR path.")
    else:
        enroll_speakers(files_map, name_mapping)
        test_accuracy(files_map, name_mapping)