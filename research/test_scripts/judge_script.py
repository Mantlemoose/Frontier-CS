import requests
import time
import os
import re
import json
import concurrent.futures
from llm_interface import *

# --- Configuration ---
K_ITERATIONS = 5  # How many times to try solving each problem
API_WORKERS = 8
JUDGE_WORKERS = 4
MODEL = 'gemini' # 'gpt' or 'gemini'
solution_dir = r'solutions'

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

    def get_result(self, sid, timeout=300, poll_interval=2):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.judge_url}/result/{sid}")
                if response.status_code == 404:
                    time.sleep( poll_interval)
                    continue
                response.raise_for_status()
                result = response.json()
                if result.get('status') in ['done', 'error']:
                    return result
                time.sleep(poll_interval)
            except requests.RequestException:
                time.sleep(poll_interval)
        return {"status": "error", "error": "Polling timed out"}

def extract_cpp_code(response_text):
    match = re.search(r'```cpp\n(.*?)```', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text

def process_single_attempt(pid, iteration):
    """
    Generates one solution, judges it, and saves the results with a unique filename
    inside a problem-specific folder.
    """
    run_id = f"{pid}_{iteration}_{MODEL}"
    print(f"[Run {iteration+1}/{K_ITERATIONS}] Starting attempt for problem: {pid}")
    
    judge = LocalJudge()
    llm = Gemini() if MODEL == 'gemini' else GPT()

    # --- MODIFICATION START ---
    # Create a problem-specific directory inside the main solutions folder
    problem_specific_dir = os.path.join(solution_dir, pid)
    os.makedirs(problem_specific_dir, exist_ok=True)

    # Update filenames to point to the new problem-specific directory
    solution_filename = os.path.join(problem_specific_dir, f"{run_id}_solution.cpp")
    result_filename = os.path.join(problem_specific_dir, f"{run_id}_result.json")
    # --- MODIFICATION END ---

    try:
        statement = judge.get_problem_statement(pid)
        if not statement: 
            print(f"[Run {iteration+1}/{K_ITERATIONS}] Failed to get statement for {pid}.")
            return

        llm_response, _ = llm.generate_solution(statement)
        if not llm_response:
            print(f"[Run {iteration+1}/{K_ITERATIONS}] LLM failed for {pid}.")
            final_result = {"status": "error", "error": "LLM_TIMEOUT_OR_FAILURE", "score": 0}
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=4)
            return

        solution_code = extract_cpp_code(llm_response)
        
        with open(solution_filename, 'w', encoding='utf-8') as f:
            f.write(solution_code)

        submission_id = judge.submit_solution(pid, solution_code)
        if not submission_id:
            final_result = {"status": "error", "error": "SUBMISSION_FAILED", "score": 0}
            with open(result_filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=4)
            return

        final_result = judge.get_result(submission_id)
        new_score = final_result.get('score', -1)
        
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4)
            
        print(f"[Run {iteration+1}/{K_ITERATIONS}] Finished attempt for {pid}. Score: {new_score}")

    except Exception as e:
        print(f"[Run {iteration+1}/{K_ITERATIONS}] A critical error occurred while processing {pid}: {e}")
        error_result = {"status": "error", "error": str(e), "score": 0}
        # Try to write error to the correct path, even if it was defined earlier
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=4)


if __name__ == "__main__":
    if not os.path.exists(solution_dir):
        os.makedirs(solution_dir)

    judge = LocalJudge()
    print("Fetching list of problems...")
    problem_ids = judge.get_all_problems()
    if not problem_ids:
        print("No problems found on the judge.")
        exit(0)
    
    print(f"Found {len(problem_ids)} problems. Starting {K_ITERATIONS} runs for each...")

    all_jobs = []
    for i in range(K_ITERATIONS):
        for pid in problem_ids:
            all_jobs.append((pid, i)) # Tuple of (problem_id, iteration_number)

    # We can use the larger API_WORKERS pool since most of the time is spent waiting.
    # The JUDGE_WORKERS limit will naturally throttle submissions.
    with concurrent.futures.ThreadPoolExecutor(max_workers=API_WORKERS) as executor:
        # Create a list of all jobs to be done
        futures = [executor.submit(process_single_attempt, pid, i) for pid, i in all_jobs]
        
        # Wait for all jobs to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result() # We can check for results or exceptions here if needed
            except Exception as e:
                print(f"A job failed in the thread pool: {e}")

    print("\nAll iterations have been processed.")