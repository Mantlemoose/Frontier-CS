#!/bin/bash
set -euo pipefail

BASE_DIR=$(pwd)

# Add user to docker group if not already a member
# Only run this on Linux systems, not Windows or macOS
# if [[ "${OSTYPE:-linux}" != "darwin"* ]] && [[ "${OSTYPE:-linux}" != "msys"* ]] && [[ "${OSTYPE:-linux}" != "cygwin"* ]] && [[ "${OSTYPE:-linux}" != "win32"* ]]; then
#     # Only run this on Linux systems, not Windows
#     if ! groups | grep -q '\bdocker\b'; then
#         echo "Adding current user to docker group..."
#         # Use $USER if available, otherwise try to get username from whoami
#         username="${USER:-$(whoami)}"
#         if command -v sudo >/dev/null 2>&1; then
#             sudo usermod -aG docker "$username"
#             echo "User added to docker group. You may need to log out and back in for changes to take effect."
#         else
#             echo "Warning: sudo not available, skipping Docker group modification"
#         fi
#         echo ""
#     fi
# fi

# Read working pairs from file
WORKING_PAIRS_FILE="$BASE_DIR/pairs.txt"
if [[ ! -f "$WORKING_PAIRS_FILE" ]]; then
    echo "ERROR: Pairs file not found: $WORKING_PAIRS_FILE"
    exit 1
fi

# Read working pairs into a temporary file
TEMP_PAIRS_FILE=$(mktemp)
while IFS= read -r line; do
    # Skip empty lines and comments (lines starting with #)
    if [ -n "$line" ] && ! echo "$line" | grep -q '^[[:space:]]*#'; then
        echo "$line" >> "$TEMP_PAIRS_FILE"
    fi
done < "$WORKING_PAIRS_FILE"

# Check if we have any working pairs
if [ ! -s "$TEMP_PAIRS_FILE" ]; then
    echo "ERROR: No valid pairs found in $WORKING_PAIRS_FILE"
    rm -f "$TEMP_PAIRS_FILE"
    exit 1
fi

RESULTS_DIR="$BASE_DIR/results"
mkdir -p "$RESULTS_DIR"

DATASETS_DIR="$BASE_DIR/datasets"
mkdir -p "$DATASETS_DIR"

# Detect GPU availability once
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_GPU="true"
else
    HAS_GPU="false"
fi

get_problem_runtime_config() {
    local problem_path="$1"
    python3 - <<'PY' "$problem_path"
import os
import sys

problem_dir = sys.argv[1]
config_path = os.path.join(problem_dir, "config.yaml")
timeout_val = ""
requires_gpu = ""

if os.path.exists(config_path):
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: try simple key-value parsing if PyYAML not available
        data = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Very basic YAML parsing for simple cases
            import re
            timeout_match = re.search(r'timeout_seconds:\s*(\d+)', content)
            if timeout_match:
                timeout_val = timeout_match.group(1)
            gpu_match = re.search(r'requires_gpu:\s*(true|false)', content, re.IGNORECASE)
            if gpu_match:
                requires_gpu = gpu_match.group(1).lower()
        except Exception:
            pass
    except Exception as exc:
        print(f"ERROR:{exc}", file=sys.stderr)
        data = {}

    if data:
        runtime = data.get("runtime") or {}
        timeout_val = runtime.get("timeout_seconds")
        requires_gpu_value = runtime.get("requires_gpu")
        if timeout_val is None:
            timeout_val = ""
        else:
            timeout_val = str(timeout_val)
        if isinstance(requires_gpu_value, bool):
            requires_gpu = "true" if requires_gpu_value else "false"
        elif requires_gpu_value is None:
            requires_gpu = ""
        else:
            requires_gpu = str(requires_gpu_value).strip().lower()
print(f"{timeout_val}|{requires_gpu}")
PY
}

echo "Starting containerized evaluation..."
echo "Working pairs: $(cat "$TEMP_PAIRS_FILE" | tr '\n' ' ')"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo "Datasets will be cached in: $DATASETS_DIR"
echo ""

# Download datasets once at the beginning (skip if already downloaded)
echo "=========================================="
echo "Downloading datasets (if not already present)..."
echo "=========================================="

# Get unique list of problems from pairs
UNIQUE_PROBLEMS=""
while IFS= read -r pair; do
    IFS=':' read -r solution problem <<< "$pair"
    # Check if problem is already in the list
    found=false
    for existing in $UNIQUE_PROBLEMS; do
        if [ "$existing" = "$problem" ]; then
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        UNIQUE_PROBLEMS="$UNIQUE_PROBLEMS $problem"
    fi
done < "$TEMP_PAIRS_FILE"

# Download datasets for each unique problem
for problem_name in $UNIQUE_PROBLEMS; do
    # Convert problem notation: imagenet_pareto/1m -> imagenet_pareto/1m or imagenet_pareto (if no slash)
    # Check both nested (with slash) and flat (without slash) formats
    problem_dir=""
    if [[ "$problem_name" == *"/"* ]]; then
        # Nested variant: imagenet_pareto/1m
        problem_dir="research/$problem_name"
    else
        # Flat: imagenet_pareto
        problem_dir="research/$problem_name"
    fi
    
    if [ ! -d "$problem_dir" ]; then
        echo "Warning: Problem '$problem_name' not found at $problem_dir, skipping dataset download..." >&2
        continue
    fi
    
    download_script="$BASE_DIR/$problem_dir/download_datasets.sh"
    
    if [ -f "$download_script" ]; then
        echo "[Download] Preparing datasets for $problem_name..." >&2
        # Redirect all output to stderr so it doesn't get captured
        if ! bash "$download_script" >&2; then
            echo "Warning: Dataset download failed for $problem_name, continuing anyway..." >&2
        fi
    fi
done

echo "=========================================="
echo "Dataset preparation complete."
echo "=========================================="
echo ""

# Function to run a single solution-problem pair in a container
run_solution_problem_pair() {
    local solution_name="$1"
    local problem_name="$2"
    # Sanitize container name by replacing slashes with underscores
    local sanitized_problem=$(echo "$problem_name" | tr '/' '_')
    local container_name="eval_${solution_name}_${sanitized_problem}_$(date +%s)"
    
    echo "==========================================" >&2
    echo "Running: $solution_name -> $problem_name" >&2
    echo "Container: $container_name" >&2
    echo "==========================================" >&2
    
    # Create a temporary directory for this evaluation
    local temp_dir=$(mktemp -d)
    
    local result_file="$RESULTS_DIR/${solution_name}_${sanitized_problem}_result.txt"
    
    # Create minimal directory structure and copy only necessary files
    mkdir -p "$temp_dir/Frontier-CS/problems"
    mkdir -p "$temp_dir/Frontier-CS/solutions"
    
    # For nested problem paths, create parent directories
    if [[ "$problem_name" == *"/"* ]]; then
        # Create full parent directory path for nested variant
        mkdir -p "$temp_dir/Frontier-CS/research/$(dirname "$problem_name")"
    fi
    
    # Copy only the specific problem and solution directories
    # For nested problems like imagenet_pareto/1m, this preserves the structure
    # Use tar to preserve directory structure properly
    (cd "$BASE_DIR/problems" && tar cf - "$problem_name" | tar xf - -C "$temp_dir/Frontier-CS/problems") || \
        cp -r "$BASE_DIR/research/$problem_name" "$temp_dir/Frontier-CS/research/"

    # Copy common/ directories from parent levels (for shared evaluator code)
    # e.g., for poc_generation/heap_buffer_overflow/arvo_47101, copy poc_generation/common/ if it exists
    local current_path=""
    IFS='/' read -ra path_parts <<< "$problem_name"
    for ((i=0; i<${#path_parts[@]}-1; i++)); do
        if [ -z "$current_path" ]; then
            current_path="${path_parts[$i]}"
        else
            current_path="$current_path/${path_parts[$i]}"
        fi
        local common_dir="$BASE_DIR/research/$current_path/common"
        if [ -d "$common_dir" ]; then
            mkdir -p "$temp_dir/Frontier-CS/research/$current_path"
            cp -r "$common_dir" "$temp_dir/Frontier-CS/research/$current_path/"
            echo "[INFO] Copied shared code from $current_path/common/" >&2
        fi
    done

    cp -r "$BASE_DIR/solutions/$solution_name" "$temp_dir/Frontier-CS/solutions/"

    local runtime_timeout=""
    local runtime_requires_gpu=""
    if runtime_config=$(get_problem_runtime_config "$BASE_DIR/research/$problem_name"); then
        IFS='|' read -r runtime_timeout runtime_requires_gpu <<< "$runtime_config"
        runtime_timeout="${runtime_timeout//[[:space:]]/}"
        runtime_requires_gpu="${runtime_requires_gpu//[[:space:]]/}"
    else
        runtime_timeout=""
        runtime_requires_gpu=""
    fi
    
    # Determine which Docker image to use based on configuration file
    local docker_image="python:3.11-slim-trixie"  # Default image
    local gpu_flags=""  # Default: no GPU flags
    local config_requests_gpu="unknown"
    if [[ "$runtime_requires_gpu" == "true" || "$runtime_requires_gpu" == "1" || "$runtime_requires_gpu" == "yes" ]]; then
        config_requests_gpu="true"
    elif [[ "$runtime_requires_gpu" == "false" || "$runtime_requires_gpu" == "0" || "$runtime_requires_gpu" == "no" ]]; then
        config_requests_gpu="false"
    fi
    
    # Docker-in-Docker flags (for mounting Docker socket)
    local dind_flags=""

    # Check if there's a docker_images.txt configuration file
    if [ -f "$BASE_DIR/docker_images.txt" ]; then
        # Extract base problem name (before any slash for nested variants)
        local base_problem=$(echo "$problem_name" | cut -d'/' -f1)

        # Look for the problem in the configuration file
        local config_line=$(grep "^$base_problem=" "$BASE_DIR/docker_images.txt")
        if [ -n "$config_line" ]; then
            # Parse the configuration line: problem_name=docker_image,gpu_support,dind
            local full_value=$(echo "$config_line" | cut -d'=' -f2 | tr -d '\r\n')
            docker_image=$(echo "$full_value" | cut -d',' -f1 | tr -d '\r\n')
            local gpu_setting=$(echo "$full_value" | cut -d',' -f2 | tr -d '\r\n')
            local dind_setting=$(echo "$full_value" | cut -d',' -f3 | tr -d '\r\n')

            # Set GPU flags based on configuration
            case "$gpu_setting" in
                "gpu"|"true"|"1")
                    gpu_flags="--gpus all"
                    echo "[INFO] Using predefined Docker image: $docker_image (with GPU support)" >&2
                    ;;
                "nogpu"|"false"|"0"|"")
                    gpu_flags=""
                    echo "[INFO] Using predefined Docker image: $docker_image (no GPU support)" >&2
                    ;;
                *)
                    gpu_flags=""
                    echo "[INFO] Using predefined Docker image: $docker_image (unknown GPU setting: '$gpu_setting', defaulting to no GPU)" >&2
                    ;;
            esac

            # Set Docker-in-Docker flags based on configuration
            case "$dind_setting" in
                "dind"|"docker")
                    dind_flags="-v /var/run/docker.sock:/var/run/docker.sock"
                    echo "[INFO] Docker-in-Docker enabled: mounting Docker socket" >&2
                    ;;
                *)
                    dind_flags=""
                    ;;
            esac
        else
            echo "[INFO] Problem not found in docker_images.txt, using default Docker image: $docker_image" >&2
        fi
    else
        echo "[INFO] No docker_images.txt found, using default Docker image: $docker_image" >&2
    fi

    if [[ "$config_requests_gpu" == "true" && -z "$gpu_flags" ]]; then
        gpu_flags="--gpus all"
        echo "[INFO] Problem runtime config requires GPU; enabling GPU flags." >&2
    elif [[ "$config_requests_gpu" == "false" && -n "$gpu_flags" ]]; then
        echo "[WARN] Problem runtime config disables GPU but docker_images.txt requested GPU; continuing with docker_images.txt setting." >&2
    fi
    
    if [[ -n "$gpu_flags" && "$HAS_GPU" != "true" ]]; then
        echo "[WARN] $solution_name requires GPU but none detected; skipping." >&2
        echo "SKIP: GPU unavailable on host" > "$result_file"
        echo "Full output saved to: $result_file" >&2
        echo "" >&2
        rm -rf "$temp_dir"
        echo "SKIP: GPU unavailable on host"
        return
    fi
    
    # Initialize exit code
    local exit_code=0
    local timed_out="false"
    local timeout_seconds=""

    if [[ -n "$runtime_timeout" ]]; then
        if [[ "$runtime_timeout" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            timeout_seconds="$runtime_timeout"
        else
            echo "[WARN] Invalid timeout value '$runtime_timeout' in $problem_name/config.yaml; ignoring." >&2
        fi
    fi
    
    if [[ -n "$timeout_seconds" ]] && ! command -v timeout >/dev/null 2>&1; then
        echo "[WARN] Timeout command not available; cannot enforce ${timeout_seconds}s limit for $problem_name." >&2
        timeout_seconds=""
    fi
    
    if [[ -n "$timeout_seconds" ]]; then
        echo "[INFO] Enforcing timeout of ${timeout_seconds}s for $solution_name -> $problem_name" >&2
        timeout --foreground "${timeout_seconds}s" docker run --rm \
            --name "$container_name" \
            $gpu_flags \
            $dind_flags \
            -v "$temp_dir:/workspace:ro" \
            -v "$DATASETS_DIR:/datasets:ro" \
            -w "/work" \
            "$docker_image" \
            bash -c '
                set -euo pipefail
                
                # Function to handle errors
                error_handler() {
                    local exit_code=$?
                    echo "ERROR: Command failed with exit code $exit_code" >&2
                    echo "ERROR_DETAILS: $(cat /tmp/error.log 2>/dev/null || echo '\''No detailed error log'\'')" >&2
                    exit $exit_code
                }
                
                # Set error trap
                trap error_handler ERR
                
                # Redirect stderr to error log for detailed error capture
                exec 2> >(tee /tmp/error.log >&2)
                
                # Copy everything from read-only mount to writable workspace
                echo "[INFO] Setting up writable workspace..."
                cp -r /workspace/Frontier-CS /work/
                cd /work/Frontier-CS
                
                # Create execution environment
                echo "[INFO] Creating execution environment..."
                mkdir -p execution_env
                
                # Set up problem environment
                echo "[INFO] Setting up problem environment for '"$problem_name"'..."
                cd research/'"$problem_name"'
                chmod +x ./set_up_env.sh 2>/dev/null || true
                if ! ./set_up_env.sh; then
                    echo "ERROR: Failed to set up problem environment"
                    exit 1
                fi
                # Return to Frontier-CS root
                cd /work/Frontier-CS
                
                # Prepare and run solution
                echo "[INFO] Preparing solution '"$solution_name"'..."
                cd solutions/'"$solution_name"'
                chmod +x ./prepare_env.sh 2>/dev/null || true
                if ! ./prepare_env.sh; then
                    echo "ERROR: Failed to prepare solution environment"
                    exit 1
                fi
                
                # Copy solution to execution environment
                echo "[INFO] Copying solution to execution environment..."
                echo "[DEBUG] mkdir -p /work/Frontier-CS/execution_env/solution_env" >&2
                mkdir -pv /work/Frontier-CS/execution_env/solution_env >&2
                echo "[DEBUG] Solution file source: /work/Frontier-CS/solutions/'"$solution_name"'/resources/solution.py" >&2
                ls -la /work/Frontier-CS/solutions/'"$solution_name"'/resources/ >&2
                echo "[DEBUG] Copy command: cp /work/Frontier-CS/solutions/'"$solution_name"'/resources/solution.py /work/Frontier-CS/execution_env/solution_env/" >&2
                if ! cp /work/Frontier-CS/solutions/'"$solution_name"'/resources/solution.py /work/Frontier-CS/execution_env/solution_env/; then
                    echo "ERROR: Failed to copy solution.py"
                    exit 1
                fi
                echo "[DEBUG] Verify copy result:" >&2
                ls -la /work/Frontier-CS/execution_env/solution_env/ >&2
                
                echo "[INFO] Running solution..."
                cd /work/Frontier-CS/solutions/'"$solution_name"'
                chmod +x ./solve.sh 2>/dev/null || true
                if ! ./solve.sh; then
                    echo "ERROR: Solution execution failed"
                    exit 1
                fi
                
                # Copy evaluate.sh to execution environment
                echo "[INFO] Setting up evaluation..."
                if ! cp /work/Frontier-CS/research/'"$problem_name"'/evaluate.sh /work/Frontier-CS/execution_env/; then
                    echo "ERROR: Failed to copy evaluate.sh"
                    exit 1
                fi
                
                # Run evaluation
                echo "[INFO] Running evaluation..."
                cd /work/Frontier-CS/research/'"$problem_name"'
                chmod +x ./evaluate.sh 2>/dev/null || true
                if ! ./evaluate.sh; then
                    echo "ERROR: Evaluation failed"
                    exit 1
                fi
            ' > "$result_file" 2>&1 || exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            timed_out="true"
        fi
    else
        docker run --rm \
            --name "$container_name" \
            $gpu_flags \
            $dind_flags \
            -v "$temp_dir:/workspace:ro" \
            -v "$DATASETS_DIR:/datasets:ro" \
            -w "/work" \
            "$docker_image" \
            bash -c '
            set -euo pipefail
            
            # Function to handle errors
            error_handler() {
                local exit_code=$?
                echo "ERROR: Command failed with exit code $exit_code" >&2
                echo "ERROR_DETAILS: $(cat /tmp/error.log 2>/dev/null || echo '\''No detailed error log'\'')" >&2
                exit $exit_code
            }
            
            # Set error trap
            trap error_handler ERR
            
            # Redirect stderr to error log for detailed error capture
            exec 2> >(tee /tmp/error.log >&2)
            
            # Copy everything from read-only mount to writable workspace
            echo "[INFO] Setting up writable workspace..."
            cp -r /workspace/Frontier-CS /work/
            cd /work/Frontier-CS
            
            # Create execution environment
            echo "[INFO] Creating execution environment..."
            mkdir -p execution_env
            
            # Set up problem environment
            echo "[INFO] Setting up problem environment for '"$problem_name"'..."
            cd research/'"$problem_name"'
            chmod +x ./set_up_env.sh 2>/dev/null || true
            if ! ./set_up_env.sh; then
                echo "ERROR: Failed to set up problem environment"
                exit 1
            fi
            # Return to Frontier-CS root
            cd /work/Frontier-CS
            
            # Prepare and run solution
            echo "[INFO] Preparing solution '"$solution_name"'..."
            cd solutions/'"$solution_name"'
            chmod +x ./prepare_env.sh 2>/dev/null || true
            if ! ./prepare_env.sh; then
                echo "ERROR: Failed to prepare solution environment"
                exit 1
            fi
            
            # Copy solution to execution environment
            echo "[INFO] Copying solution to execution environment..."
            echo "[DEBUG] mkdir -p /work/Frontier-CS/execution_env/solution_env" >&2
            mkdir -pv /work/Frontier-CS/execution_env/solution_env >&2
            echo "[DEBUG] Solution file source: /work/Frontier-CS/solutions/'"$solution_name"'/resources/solution.py" >&2
            ls -la /work/Frontier-CS/solutions/'"$solution_name"'/resources/ >&2
            echo "[DEBUG] Copy command: cp /work/Frontier-CS/solutions/'"$solution_name"'/resources/solution.py /work/Frontier-CS/execution_env/solution_env/" >&2
            if ! cp /work/Frontier-CS/solutions/'"$solution_name"'/resources/solution.py /work/Frontier-CS/execution_env/solution_env/; then
                echo "ERROR: Failed to copy solution.py"
                exit 1
            fi
            echo "[DEBUG] Verify copy result:" >&2
            ls -la /work/Frontier-CS/execution_env/solution_env/ >&2
            
            echo "[INFO] Running solution..."
            cd /work/Frontier-CS/solutions/'"$solution_name"'
            chmod +x ./solve.sh 2>/dev/null || true
            if ! ./solve.sh; then
                echo "ERROR: Solution execution failed"
                exit 1
            fi
            
            # Copy evaluate.sh to execution environment
            echo "[INFO] Setting up evaluation..."
            if ! cp /work/Frontier-CS/research/'"$problem_name"'/evaluate.sh /work/Frontier-CS/execution_env/; then
                echo "ERROR: Failed to copy evaluate.sh"
                exit 1
            fi
            
            # Run evaluation
            echo "[INFO] Running evaluation..."
                cd /work/Frontier-CS/research/'"$problem_name"'
                chmod +x ./evaluate.sh 2>/dev/null || true
                if ! ./evaluate.sh; then
                    echo "ERROR: Evaluation failed"
                    exit 1
                fi
            ' > "$result_file" 2>&1 || exit_code=$?
    fi
    
    # Extract score from result file and handle errors
    local score
    # Try to extract the last line as score, filtering out log messages
    local last_line=$(tail -1 "$result_file" 2>/dev/null || echo "")
    
    # Check if last line looks like a number (score) and is not a log message
    if echo "$last_line" | grep -q '^-?[0-9]\+\.\?[0-9]*$' && ! echo "$last_line" | grep -q '\[INFO\]'; then
        score="$last_line"
    elif [ $exit_code -eq 0 ]; then
        # Success case but no numeric score - look for any numeric line that's not a log message
        local numeric_line=$(grep -E '^-?[0-9]+\.?[0-9]*$' "$result_file" | grep -v '\[INFO\]' | tail -1)
        if [ -n "$numeric_line" ]; then
            score="$numeric_line"
        else
            score="$last_line"
        fi
    else
        # Error case - extract detailed error message
        local error_msg=""
        if [[ "$timed_out" == "true" ]]; then
            local display_timeout="$timeout_seconds"
            if [[ -z "$display_timeout" && -n "$runtime_timeout" ]]; then
                display_timeout="$runtime_timeout"
            fi
            score="ERROR: Execution timed out after ${display_timeout}s"
        elif grep -Eq "[A-Za-z]+Error:" "$result_file"; then
            error_msg=$(grep -E "[A-Za-z]+Error:" "$result_file" | head -1)
        elif grep -q "ERROR:" "$result_file"; then
            error_msg=$(grep "ERROR:" "$result_file" | tail -1)
        else
            # Look for actual error patterns in the last 50 lines
            error_msg=$(tail -50 "$result_file" | grep -iE "(error|failed|exception|traceback)" | tail -1)
        fi
        
        # Clean up the error message
        if [ -n "$error_msg" ]; then
            score="ERROR: $error_msg"
        else
            score="ERROR: Container execution failed (exit code: $exit_code)"
        fi
    fi
    
    echo "Full output saved to: $result_file" >&2
    echo "" >&2
    
    # Clean up temp directory
    rm -rf "$temp_dir"
    
    # Return the score for summary
    echo "$score"
}

# Initialize results summary
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "Evaluation Summary - $(date)" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Initialize CSV file
CSV_FILE="$RESULTS_DIR/results.csv"
echo "solution,problem,score,status,timestamp" > "$CSV_FILE"

# Function to escape CSV fields (handles commas, quotes, newlines)
escape_csv_field() {
    local field="$1"
    # Replace quotes with double quotes and wrap in quotes if needed
    if [[ "$field" == *[\",\n]* ]]; then
        field="${field//\"/\"\"}"
        echo "\"$field\""
    else
        echo "$field"
    fi
}

# Run only the working solution-problem pairs
while IFS= read -r pair; do
    IFS=':' read -r solution problem <<< "$pair"
    
    # Check if problem and solution exist
    if [ ! -d "research/$problem" ]; then
        echo "ERROR: Problem '$problem' not found, skipping..."
        echo "$solution -> $problem: ERROR (problem not found)" >> "$SUMMARY_FILE"
        # Write to CSV
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$solution,$problem,N/A,ERROR: problem not found,$timestamp" >> "$CSV_FILE"
        continue
    fi

    if [ ! -d "solutions/$solution" ]; then
        echo "ERROR: Solution '$solution' not found, skipping..."
        echo "$solution -> $problem: ERROR (solution not found)" >> "$SUMMARY_FILE"
        # Write to CSV
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$solution,$problem,N/A,ERROR: solution not found,$timestamp" >> "$CSV_FILE"
        continue
    fi

    # Run the evaluation
    score=$(run_solution_problem_pair "$solution" "$problem")
    echo "$solution -> $problem:" >> "$SUMMARY_FILE"
    echo "Score: $score" >> "$SUMMARY_FILE"

    # Write to CSV
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    if [[ "$score" == ERROR:* ]]; then
        # Extract just the error message for the status column
        error_msg="${score#ERROR: }"
        escaped_error=$(escape_csv_field "$error_msg")
        echo "$solution,$problem,N/A,\"ERROR: $escaped_error\",$timestamp" >> "$CSV_FILE"
    else
        # Numeric score - treat as success
        escaped_score=$(escape_csv_field "$score")
        echo "$solution,$problem,$escaped_score,SUCCESS,$timestamp" >> "$CSV_FILE"
    fi
done < "$TEMP_PAIRS_FILE"

# Clean up temporary file
rm -f "$TEMP_PAIRS_FILE"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Summary saved to: $SUMMARY_FILE"
echo "CSV results saved to: $CSV_FILE"
echo ""
echo "Full summary:"
cat "$SUMMARY_FILE"
echo ""
echo "CSV preview (first 10 lines):"
head -10 "$CSV_FILE"

# Signal completion atomically for remote fetchers
touch "$RESULTS_DIR/.done" 2>/dev/null || true
