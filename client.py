"""
Script Name: client.py
Description: Test script for the IAMI Compliant Speaker Identification API.
Features: Explicit Enroll vs Probe tasks, Attribute Printing, Identify, and Resolve.

Date: 10/12/2025
"""
import os
import json
import argparse
import requests
import uuid
from datetime import datetime, timezone
from collections import defaultdict

BASE_URL = "http://localhost:8006"

# Directories
IAMI_TEST_FILES_ENROLL = "/home/lorenzo-concina/development/datasets/Voice_recording_three_suspects/Audio_to_enroll"
IAMI_TEST_FILES_PROBE = "/home/lorenzo-concina/development/datasets/Voice_recording_three_suspects/Audio_probe_1"

def collect_audio_files(root_dir):
    """Collects all .wav/.mp3 files from a directory and groups them by user ID (prefix)."""
    user_audio_map = defaultdict(list)
    try:
        for filename in os.listdir(root_dir):
            if filename.lower().endswith((".wav", ".mp3")):
                # Be careful with split if filename doesn't have underscore
                parts = filename.split("_")
                speaker_id = parts[0] if len(parts) > 0 else "unknown"
                full_path = os.path.join(root_dir, filename)
                user_audio_map[speaker_id].append(full_path)
    except FileNotFoundError:
        print(f"Warning: Directory not found at {root_dir}")
    return dict(user_audio_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client for the IAMI Speaker Identification API')
    
    parser.add_argument('--task', required=True, choices=[
        'check_connection', 'extract_attributes', 'extract_attributes_probe', 'identify', 'resolve', 'list_users', 'delete_user_embeddings'
    ], help='Task to be performed.')
    
    parser.add_argument('--userid', required=False, help='User ID required for deleting a user.')
    parser.add_argument('--query_id', required=False, help='A specific ChromaDB ID to identify or resolve.')
    parser.add_argument('--session_id', required=False, help='The session ID context for the probe.')

    args = parser.parse_args()
    
    try:
        if args.task == "check_connection":
            print("Checking server connection...")
            response = requests.get(f"{BASE_URL}/speaker-ID/")
            print(response.json())

        # ==========================================
        # TASK 1: EXTRACT ATTRIBUTES (ENROLL)
        # ==========================================
        elif args.task == "extract_attributes":
            print(f"\n--- Starting Enrollment from {IAMI_TEST_FILES_ENROLL} ---")
            files_map = collect_audio_files(IAMI_TEST_FILES_ENROLL)
            
            if not files_map:
                print("No files found.")
                exit()

            for user_id, file_paths in files_map.items():
                print(f"\n Uploading {len(file_paths)} files for User: {user_id}...")

                req_data_dict = {
                    "case_id": user_id,
                    "session_id": str(uuid.uuid4()), 
                    "request_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "return_debug_info": True
                }
                
                # --- CONSTRUCTION OF MULTIPART PAYLOAD ---
                multipart_payload = [
                    ('request_data', (None, json.dumps(req_data_dict))), 
                ]
                
                opened_files = []
                try:
                    for fpath in file_paths:
                        f = open(fpath, "rb")
                        opened_files.append(f)
                        multipart_payload.append(('audios', (os.path.basename(fpath), f, 'audio/wav')))

                    response = requests.post(
                        f"{BASE_URL}/speaker-ID/extract_attributes",
                        files=multipart_payload
                    )
                    
                    if response.status_code != 200:
                        print(f"❌ Error {response.status_code}: {response.text}")
                        response.raise_for_status()

                    resp_json = response.json()

                    if resp_json.get("success"):
                        print(f"    Success")
                        for entry in resp_json.get("entries", []):
                            print(f"       ID: {entry.get('collection_db_entry_id')}")
                            if entry.get("attributes"):
                                print("       Attributes:")
                                for attr in entry["attributes"]:
                                    name = attr.get("type", {}).get("name", "unknown")
                                    val = attr.get("values", ["N/A"])[0]
                                    conf = attr.get("confidences", [0.0])[0]
                                    
                                    # Format confidence to 3 decimal places if it's a float
                                    conf_str = f"{conf:.3f}" if isinstance(conf, float) else conf
                                    print(f"         - {name}: {val} (Confidence: {conf_str})")
                    else:
                        print(f"    Failed: {resp_json.get('message')}")

                finally:
                    for f in opened_files: f.close()

        # ==========================================
        # TASK 2: EXTRACT ATTRIBUTES PROBE
        # ==========================================
        elif args.task == "extract_attributes_probe":
            print(f"\n--- Starting Probe Extraction from {IAMI_TEST_FILES_PROBE} ---")
            files_map = collect_audio_files(IAMI_TEST_FILES_PROBE)
            
            if not files_map:
                print("No files found.")
                exit()

            shared_session_id = str(uuid.uuid4())
            print(f"ℹ  Using Shared Session ID: {shared_session_id}")

            for user_id, file_paths in files_map.items():
                print(f"\n Uploading {len(file_paths)} files for User: {user_id}...")

                req_data_dict = {
                    "case_id": user_id,
                    "session_id": shared_session_id, 
                    "request_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "return_debug_info": True
                }
                
                # --- UPDATED MULTIPART PAYLOAD ---
                multipart_payload = [
                    ('request_data', (None, json.dumps(req_data_dict))), 
                ]
                
                opened_files = []
                try:
                    for fpath in file_paths:
                        f = open(fpath, "rb")
                        opened_files.append(f)
                        multipart_payload.append(('audios', (os.path.basename(fpath), f, 'audio/wav')))

                    response = requests.post(
                        f"{BASE_URL}/speaker-ID/extract_attributes_probe",
                        files=multipart_payload
                    )
                    
                    if response.status_code != 200:
                        print(f"❌ Error {response.status_code}: {response.text}")
                        response.raise_for_status()

                    resp_json = response.json()

                    if resp_json.get("success"):
                        print(f"    Success")
                        for entry in resp_json.get("entries", []):
                            print(f"       ID: {entry.get('collection_db_entry_id')}")
                            if entry.get("attributes"):
                                print("       Attributes:")
                                for attr in entry["attributes"]:
                                    name = attr.get("type", {}).get("name", "unknown")
                                    val = attr.get("values", ["N/A"])[0]
                                    conf = attr.get("confidences", [0.0])[0]
                                    
                                    # Format confidence to 3 decimal places if it's a float
                                    conf_str = f"{conf:.3f}" if isinstance(conf, float) else conf
                                    print(f"         - {name}: {val} (Confidence: {conf_str})")
                    else:
                        print(f"    Failed: {resp_json.get('message')}")

                finally:
                    for f in opened_files: f.close()

        # ==========================================
        # TASK 3: IDENTIFY
        # ==========================================
        elif args.task == "identify":
            if not args.query_id:
                print(" Error: --query_id is required.")
                exit()

            current_session = args.session_id if args.session_id else str(uuid.uuid4())
            anchor_uuid = str(uuid.uuid4())

            identify_payload = {
                "case_id": "manual_test",
                "session_id": current_session,
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
                "probe": {
                    "anchor_entry_id": anchor_uuid,
                    "entities": [
                        {
                            "id": anchor_uuid,
                            "collection_db_entry_id": args.query_id,
                            "collection": "natural_person",
                            "attributes": []
                        }
                    ]
                },
                "max_results": 5
            }

            print(f"\n🔍 Identifying ID: {args.query_id} (Session: {current_session})")
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                f"{BASE_URL}/speaker-ID/identify",
                headers=headers,
                data=json.dumps(identify_payload)
            )
            
            if response.status_code != 200:
                 print(f"❌ Error {response.status_code}: {response.text}")
                 response.raise_for_status()

            resp_json = response.json()
            
            print("\n--- RESULTS ---")
            if resp_json.get("success"):
                matches = resp_json.get("matches", [])
                if not matches:
                    print("   No matches found.")
                for m in matches:
                    score = m.get("final_score")
                    candidate = m.get("candidate_entry", {})
                    group = candidate.get("group_id")
                    db_id = candidate.get("collection_db_entry_id")
                    reasoning = m.get("reasoning", [""])[0]
                    
                    print(f"    User: {group} | Score: {score:.4f} | ID: {db_id}")
                    print(f"       {reasoning}")
                    print("-" * 30)
            else:
                print(f" Error: {resp_json.get('message')}")

        # ==========================================
        # TASK 4: RESOLVE (New Functionality)
        # ==========================================
        elif args.task == "resolve":
            if not args.query_id:
                print(" Error: --query_id is required.")
                exit()
            if not args.session_id:
                print(" Error: --session_id is required for Resolution.")
                exit()

            current_session = args.session_id
            anchor_uuid = str(uuid.uuid4())

            resolve_payload = {
                "case_id": "manual_test",
                "session_id": current_session,
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
                "probe": {
                    "anchor_entry_id": anchor_uuid,
                    "entities": [
                        {
                            "id": anchor_uuid,
                            "collection_db_entry_id": args.query_id,
                            "collection": "natural_person",
                            "attributes": []
                        }
                    ]
                },
                "max_results": 5
            }

            print(f"\n🔍 Resolving ID: {args.query_id} (Session: {current_session})")
            
            headers = {"Content-Type": "application/json"}
            # Notice the endpoint change here
            response = requests.post(
                f"{BASE_URL}/speaker-ID/resolve",
                headers=headers,
                data=json.dumps(resolve_payload)
            )
            
            if response.status_code != 200:
                 print(f"❌ Error {response.status_code}: {response.text}")
                 response.raise_for_status()

            resp_json = response.json()
            
            print("\n--- RESOLUTION RESULTS ---")
            if resp_json.get("success"):
                matches = resp_json.get("matches", [])
                if not matches:
                    print("   No local matches found in this session.")
                for m in matches:
                    score = m.get("final_score")
                    candidate = m.get("candidate_entry", {})
                    # group might be 'unknown' or a temporary label in probe logic
                    group = candidate.get("group_id")
                    db_id = candidate.get("collection_db_entry_id")
                    reasoning = m.get("reasoning", [""])[0]
                    
                    print(f"    Match | Score: {score:.4f} | ID: {db_id}")
                    print(f"       {reasoning}")
                    print("-" * 30)
            else:
                print(f" Error: {resp_json.get('message')}")

        elif args.task == "list_users":
            response = requests.get(f"{BASE_URL}/speaker-ID/users")
            print("Users:", response.json().get("users", []))

        elif args.task == "delete_user_embeddings":
            if not args.userid: raise ValueError("User ID required")
            requests.request("DELETE", f"{BASE_URL}/speaker-ID/user/delete", json={"user_id": args.userid})
            print(f"Deleted {args.userid}")

    except Exception as e:
        print(f"Error: {e}")