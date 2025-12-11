import requests
import time
import os
import re
import json
import concurrent.futures
import argparse
import sys
import threading
from llm_interface import *

# --- Configuration ---
API_WORKERS = 16    # Max concurrent LLM generations
JUDGE_WORKERS = 8   # Max concurrent submissions to the local judge
# Get the directory where this script is currently located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set solution_dir to be a folder named 'solutions' inside that directory
solution_dir = os.path.join(script_dir, 'solutions')

# Define available models here for validation
AVAILABLE_MODELS = {
    'gemini', 'gpt', 'claude', 'claude-opus', 
    'claude-sonnet-4-5', 'gemini3', 'Grok', 'claude-opus-4-5'
}

# Semaphore to control concurrent access to the judge
judge_throttler = threading.BoundedSemaphore(JUDGE_WORKERS)

class LocalJudge:
    def __init__(self, judge_url="http://localhost:8081"):
        self.judge_url = judge_url
        self.session = requests.Session()

    def get_all_problems(self):
        try:
            response = self.session.get(f"{self.judge_url}/problems")
            response.raise_for_status()
            return [p['id'] for p in response.json().get('problems', [])]
        except requests.RequestException:
            print(f"Error connecting to the judge at {self.judge_url}. Is it running?")
            return None

    def get_problem_statement(self, pid):
        try:
            response = self.session.get(f"{self.judge_url}/problem/{pid}/statement")
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def submit_solution(self, pid, code):
        files = {'code': ('solution.cpp', code)}
        data = {'pid': pid, 'lang': 'cpp'}
        try:
            response = self.session.post(f"{self.judge_url}/submit", files=files, data=data)
            response.raise_for_status()
            return response.json().get('sid')
        except requests.RequestException:
            return None

    def get_result(self, sid, poll_interval=60):
        # Polls indefinitely until a result is returned
        while True:
            try:
                response = self.session.get(f"{self.judge_url}/result/{sid}")
                if response.status_code == 404:
                    time.sleep(poll_interval)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if result.get('status') in ['done', 'error']:
                    return result
                
                time.sleep(poll_interval)
                
            except requests.RequestException:
                time.sleep(poll_interval)

def extract_cpp_code(response_text):
    match = re.search(r'```cpp\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text

def get_llm_instance(model_name):
    """Factory function to instantiate the correct LLM class."""
    if model_name == 'gemini':
        return Gemini()
    elif model_name == 'gpt':
        return GPT()
    elif model_name == 'claude':
        return Claude()
    elif model_name == 'claude-opus':
        return Claude_Opus()
    elif model_name == 'claude-sonnet-4-5':
        return Claude_Sonnet_4_5()
    elif model_name == 'gemini3':
        return Gemini3()
    elif model_name == 'Grok':
        return Grok()
    elif model_name == 'claude-opus-4-5':
        return Claude_Opus_4_5()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def process_single_attempt(pid, model_name):
    """
    Generates one solution, judges it, and saves the results.
    Returns the score (float/int) for the attempt. 
    Returns 0 if a critical error occurs before grading.
    """
    run_id = f"{pid}_{model_name}"
    result_filename = f"{solution_dir}/{run_id}_result.json"
    solution_filename = f"{solution_dir}/{run_id}_solution.cpp"

    print(f"[Processing {pid}] Starting attempt using {model_name}")
    
    judge = LocalJudge()
    
    try:
        llm = get_llm_instance(model_name)
    except ValueError as e:
        print(e)
        return 0

    try:
        # 1. Get Statement
        statement = judge.get_problem_statement(pid)
        if not statement: 
            print(f"[Processing {pid}] Failed to get statement.")
            return 0

        # 2. Generate Solution
        llm_response, _ = llm.generate_solution(statement)
        if not llm_response:
            print(f"[Processing {pid}] LLM failed.")
            final_result = {"status": "error", "error": "LLM_TIMEOUT_OR_FAILURE", "score": 0}
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=4)
            return 0

        llm_text_content = str(llm_response)
        solution_code = extract_cpp_code(llm_text_content)
        
        with open(solution_filename, 'w', encoding='utf-8') as f:
            f.write(solution_code)

        # 3. Submit and Grade
        with judge_throttler:
            submission_id = judge.submit_solution(pid, solution_code)
            if not submission_id:
                final_result = {"status": "error", "error": "SUBMISSION_FAILED", "score": 0}
                with open(result_filename, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, indent=4)
                return 0

            final_result = judge.get_result(submission_id)
        
        # 4. Save Result
        new_score = final_result.get('score', -1)
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4)
            
        print(f"[Processing {pid}] Finished. Score: {new_score}")
        
        return new_score

    except Exception as e:
        print(f"[Processing {pid}] A critical error occurred: {e}")
        error_result = {"status": "error", "error": str(e), "score": 0}
        try:
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=4)
        except Exception as e_file:
            print(f"  ...Additionally, FAILED to write error log: {e_file}")
        return 0


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run LLM Judge Benchmarks")
    parser.add_argument(
        "model", 
        type=str, 
        help=f"The model to run. Options: {', '.join(AVAILABLE_MODELS)}"
    )
    args = parser.parse_args()
    
    selected_model = args.model

    if selected_model not in AVAILABLE_MODELS:
        print(f"Error: '{selected_model}' is not a valid model.")
        print(f"Available options: {', '.join(AVAILABLE_MODELS)}")
        sys.exit(1)

    if not os.path.exists(solution_dir):
        os.makedirs(solution_dir)

    judge = LocalJudge()
    print("Fetching list of problems...")
    problem_ids = judge.get_all_problems()
    if not problem_ids:
        print("No problems found on the judge.")
        sys.exit(0)
    
    print(f"Found {len(problem_ids)} problems. Starting single run for each using model: {selected_model}")

    collected_scores = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=API_WORKERS) as executor:
        futures = [executor.submit(process_single_attempt, pid, selected_model) for pid in problem_ids]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                # Retrieve the return value from the function (the score)
                raw_score = future.result()
                
                # Treat -1 (error/missing) as 0 for the average
                if raw_score == -1:
                    effective_score = 0
                else:
                    effective_score = raw_score
                
                collected_scores.append(effective_score)
                
            except Exception as e:
                print(f"A job failed in the thread pool: {e}")
                collected_scores.append(0) # Treat exceptions as 0 score

    print("\nAll problems have been processed.")
    
    # Calculate Average
    if collected_scores:
        avg_score = sum(collected_scores) / len(collected_scores)
        print("-" * 30)
        print(f"Final Average Score: {avg_score:.2f}")
        print("-" * 30)
    else:
        print("No scores collected.")