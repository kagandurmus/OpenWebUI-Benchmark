import csv
import json
import re
import statistics
import time
import uuid
import ast
import yaml
from openai import OpenAI
import re
import yaml


def load_api_key(filepath="secrets.txt"):
    with open(filepath, "r") as file:
        for line in file:
            if line.strip().startswith("API_KEY"):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise ValueError("API_KEY not found")


def load_base_url(filepath="secrets.txt"):
    with open(filepath, "r") as file:
        for line in file:
            if line.strip().startswith("BASE_URL"):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise ValueError("BASE_URL not found")


BASE_URL = load_base_url()
API_KEY = load_api_key()
MODEL = "google/gemma-4-26B-A4B-it"          # openai/gpt-oss-120b || "ibm-granite/granite-4.2-8b" || "Qwen3/Qwen3" || "google/gemma-4-26B-A4B-it"
CSV_OUTPUT_FILE = "benchmark_eval_results.csv"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# =========================================================================
# PROGRAMMATIC EVALUATORS

def sanitize_text(text: str) -> str:
    if not text: return text
    return text.replace('\xa0', ' ').replace('\u2011', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')

def eval_q1_negative_constraint(response: str) -> dict:
    """
    Q1: Negative Constraint check (Podman Ollama deployment)
    - Forbidden words: 'docker', 'virtualization', 'cloud' (case-insensitive)
    - Required concepts: 'podman', '-p' or 'port', '-d' or 'detached', '-v' or 'volume'
    """
    response_lower = sanitize_text(response).lower()
    forbidden = ["docker", "virtualization", "cloud"]
    found_forbidden = [w for w in forbidden if w in response_lower]
    
    required_keywords = ["podman"]
    missing_required = [w for w in required_keywords if w not in response_lower]
    
    has_flags = any(flag in response_lower for flag in ["-d", "-p", "-v", "--name", "port", "volume"])
    
    passed = (len(found_forbidden) == 0) and (len(missing_required) == 0) and has_flags
    
    feedback = []
    if found_forbidden:
        feedback.append(f"Used forbidden words: {found_forbidden}")
    if missing_required:
        feedback.append(f"Missing required keywords: {missing_required}")
    if not has_flags:
        feedback.append("Missing container flags/concepts (-p, -d, -v, volume, port)")
    if passed:
        feedback.append("Successfully satisfied negative constraints and described Podman deployment.")

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "feedback": " | ".join(feedback)
    }


def eval_q2_troubleshooting_command(response: str) -> dict:
    """
    Q2: Fix line ending error (/bin/bash^M: Defekter Interpreter)
    - Must mention CRLF / Windows line endings / carriage return
    - Must contain sed command: sed -i 's/\r$//' or equivalent regex for removing \r
    """
    response_lower = sanitize_text(response).lower()
    has_explanation = any(term in response_lower for term in ["crlf", "windows", "carriage return", "line ending", "linebreak", "\\r", "^m"])
    has_sed = "sed" in response_lower and ("'s/\\r$//'" in response_lower or "s/\\r$//" in response_lower or "'s/\\r//'" in response_lower or "dos2unix" in response_lower)

    passed = has_explanation and has_sed
    score = (0.5 if has_explanation else 0.0) + (0.5 if has_sed else 0.0)
    
    feedback = []
    if not has_explanation:
        feedback.append("Missing explanation of Windows CRLF / line ending issue.")
    if not has_sed:
        feedback.append("Missing correct `sed -i 's/\\r$//'` command or `dos2unix` fix.")
    if passed:
        feedback.append("Accurately explained CRLF cause and provided the exact fixing command.")

    return {
        "passed": passed,
        "score": score,
        "feedback": " | ".join(feedback)
    }


def eval_q3_strict_yaml_config(response: str) -> dict:
    """
    Q3: VS Code Continue config.yaml validation (Strict YAML)
    - Must NOT be JSON format (rejects { } and ```json).
    - Must parse as valid YAML.
    - Must contain 'models' list.
    - Model entry must have provider ('ollama'), model ('granite4'), apiBase.
    """
    # 1. Reject if it explicitly outputs a JSON code block
    if "```json" in sanitize_text(response).lower():
        return {"passed": False, "score": 0.0, "feedback": "Failed: Model output JSON block instead of requested YAML."}
        
    # Extract YAML block if enclosed in ```yaml ... ```
    match = re.search(r"```(?:yaml)?\n(.*?\n)```", sanitize_text(response), re.DOTALL | re.IGNORECASE)
    yaml_str = match.group(1) if match else response
    
    # 2. Reject if the extracted string is formatted as a JSON object with braces
    clean_str = yaml_str.strip()
    if clean_str.startswith("{") and clean_str.endswith("}"):
        return {"passed": False, "score": 0.0, "feedback": "Failed: Output uses JSON braces instead of standard YAML structure."}
        
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict) or "models" not in data:
            return {"passed": False, "score": 0.0, "feedback": "YAML parsed, but missing root key 'models'"}
        
        models = data["models"]
        if not isinstance(models, list) or len(models) == 0:
            return {"passed": False, "score": 0.0, "feedback": "'models' key is not a non-empty list"}
        
        target = None
        for m in models:
            if isinstance(m, dict) and m.get("provider") == "ollama":
                target = m
                break
        
        if not target:
            target = models[0] if isinstance(models[0], dict) else {}
            
        has_provider = target.get("provider") == "ollama"
        has_model = "granite" in str(target.get("model")).lower()
        has_apibase = "5002" in str(target.get("apiBase"))
        
        # REMOVED: has_name check, as it was never requested in the prompt.
        
        passed = has_provider and has_model and has_apibase
        score = sum([has_provider, has_model, has_apibase]) / 3.0
        
        return {
            "passed": passed,
            "score": score,
            "feedback": f"Strict YAML check - provider: {has_provider}, model: {has_model}, apiBase: {has_apibase}"
        }
    except Exception as e:
        return {"passed": False, "score": 0.0, "feedback": f"Failed to parse YAML: {str(e)}"}


def eval_q4_architecture_json(response: str) -> dict:
    """
    Q4: 3-Layer Architecture Design (Structural Test)
    - Must parse as valid JSON.
    - Must contain exactly three keys representing the tiers: 'tier_1', 'tier_2', 'tier_3'.
    """
    match = re.search(r"```(?:json)?\n(.*?\n)```", sanitize_text(response), re.DOTALL)
    json_str = match.group(1) if match else response
    
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return {"passed": False, "score": 0.0, "feedback": "Response is not a valid JSON dictionary."}
            
        required_keys = {"tier_1", "tier_2", "tier_3"}
        has_all_keys = required_keys.issubset(data.keys())
        
        # Bonus: check if they actually put text inside those keys
        has_content = all(isinstance(data.get(k), str) and len(data.get(k)) > 5 for k in required_keys) if has_all_keys else False
        
        passed = has_all_keys and has_content
        score = 1.0 if passed else (0.5 if has_all_keys else 0.0)
        
        return {
            "passed": passed,
            "score": score,
            "feedback": f"Keys present: {has_all_keys}, Values contain descriptions: {has_content}"
        }
    except Exception as e:
        return {"passed": False, "score": 0.0, "feedback": f"Failed to parse JSON: {str(e)}"}


def eval_q5_code_is_prime(response: str) -> dict:
    """
    Q5: Code Generation - is_prime(n: int) -> bool
    - Extracts python code
    - Checks for docstring
    - Executes test cases in safe sandbox
    """
    match = re.search(r"```(?:python)?\n(.*?\n)```", sanitize_text(response), re.DOTALL)
    code = match.group(1) if match else response
    
    try:
        parsed = ast.parse(code)
        func_node = None
        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef) and node.name == "is_prime":
                func_node = node
                break
                
        if not func_node:
            return {"passed": False, "score": 0.0, "feedback": "Function `def is_prime` not found in response."}
            
        docstring = ast.get_docstring(func_node)
        has_docstring = bool(docstring and len(docstring.strip()) > 0)
        
        namespace = {}
        exec(compile(parsed, filename="<ast>", mode="exec"), namespace)
        is_prime_fn = namespace.get("is_prime")
        
        test_results = [
            is_prime_fn(2) == True,
            is_prime_fn(3) == True,
            is_prime_fn(4) == False,
            is_prime_fn(1) == False,
            is_prime_fn(13) == True,
            is_prime_fn(25) == False
        ]
        
        all_tests_passed = all(test_results)
        score = (0.3 if has_docstring else 0.0) + (0.7 if all_tests_passed else 0.0)
        passed = has_docstring and all_tests_passed
        
        return {
            "passed": passed,
            "score": round(score, 2),
            "feedback": f"Docstring present: {has_docstring}, Test cases passed: {sum(test_results)}/{len(test_results)}"
        }
    except Exception as e:
        return {"passed": False, "score": 0.0, "feedback": f"Code parsing/execution error: {str(e)}"}


def eval_q6_nl_to_sql_flow(response: str) -> dict:
    """
    Q6: Database NL-to-SQL end-to-end data flow
    - Must mention sequence: User Query -> Prompt/LLM -> SQL generation -> Database Execution -> Response
    """
    resp_lower = sanitize_text(response).lower()
    has_prompt_llm = any(term in resp_lower for term in ["prompt", "llm", "granite", "ollama", "natural language"])
    has_sql_db = any(term in resp_lower for term in ["sql", "database", "query", "hana", "db"])
    has_flow_step = any(term in resp_lower for term in ["presentation", "service", "connector", "flow", "step", "layer"])
    
    passed = has_prompt_llm and has_sql_db and has_flow_step
    score = (0.33 if has_prompt_llm else 0) + (0.33 if has_sql_db else 0) + (0.34 if has_flow_step else 0)
    
    return {
        "passed": passed,
        "score": round(score, 2),
        "feedback": f"Sequence elements - Prompt/LLM: {has_prompt_llm}, SQL/DB: {has_sql_db}, Layered flow: {has_flow_step}"
    }


def eval_q7_data_standardization_workflow(response: str) -> dict:
    """
    Q7: 4-Step Address Standardization Workflow
    - Checks 4 steps: 1) Consent/Confirmation, 2) Iterating over records, 3) Standardizing via LLM, 4) Updating/Writing DB
    """
    resp_lower = sanitize_text(response).lower()
    has_consent = any(term in resp_lower for term in ["consent", "confirm", "user", "ask", "permission", "automatically"])
    has_loop = any(term in resp_lower for term in ["loop", "iterate", "each", "fetch", "records", "addresses"])
    has_llm_format = any(term in resp_lower for term in ["format", "standardize", "llm", "reformat", "clean"])
    has_db_save = any(term in resp_lower for term in ["save", "update", "write", "insert", "db", "database"])
    
    count = sum([has_consent, has_loop, has_llm_format, has_db_save])
    score = count / 4.0
    passed = (count >= 3)
    
    return {
        "passed": passed,
        "score": score,
        "feedback": f"Workflow steps check - Consent: {has_consent}, Loop: {has_loop}, LLM reformat: {has_llm_format}, DB update: {has_db_save}"
    }


def eval_q8_structured_json_slogan(response: str) -> dict:
    """
    Q8: Structured JSON Slogan Generation
    - Must parse as valid JSON
    - Must contain exact keys 'product_name', 'price', 'generated_slogan'
    """
    # Attempt 1: Extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE)
    json_str = match.group(1) if match else response
    
    # Attempt 2: Fallback to finding outermost curly braces
    if not match:
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}")
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = json_str[start_idx : end_idx + 1]

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return {"passed": False, "score": 0.0, "feedback": "JSON parsed but is not a dictionary object."}
            
        has_prod = "product_name" in data
        has_price = "price" in data
        has_slogan = "generated_slogan" in data
        slogan_non_empty = bool(str(data.get("generated_slogan", "")).strip())
        
        passed = has_prod and has_price and has_slogan and slogan_non_empty
        score = sum([has_prod, has_price, has_slogan, slogan_non_empty]) / 4.0
        
        return {
            "passed": passed,
            "score": score,
            "feedback": f"JSON schema check - product_name: {has_prod}, price: {has_price}, slogan: {has_slogan}, non-empty: {slogan_non_empty}"
        }
    except Exception as e:
        return {"passed": False, "score": 0.0, "feedback": f"Failed to parse JSON: {str(e)}"}

def eval_q9_prompt_engineering(response: str) -> dict:
    """
    Q9: Zero-Shot vs Few-Shot Prompting comparison
    - Checks definitions of Zero-Shot (no examples) and Few-Shot (in-context examples)
    - Checks reasoning why Few-Shot improves Coder-LLMs (pattern guidance, docstrings, formatting)
    """
    # Replace non-breaking hyphens and en-dashes with standard hyphens
    resp_lower = sanitize_text(response).lower().replace("‑", "-").replace("–", "-")
    # Remove hyphens entirely for a secondary check
    resp_clean = resp_lower.replace("-", "")
    has_zero = "zero-shot" in resp_lower or "zero shot" in resp_lower or "no example" in resp_lower
    has_few = "few-shot" in resp_lower or "few shot" in resp_lower or "example" in resp_lower
    has_reasoning = any(term in resp_lower for term in ["docstring", "pattern", "consistency", "structure", "autocomplete", "quality", "syntax"])
    
    passed = has_zero and has_few and has_reasoning
    score = (0.33 if has_zero else 0) + (0.33 if has_few else 0) + (0.34 if has_reasoning else 0)
    
    return {
        "passed": passed,
        "score": round(score, 2),
        "feedback": f"Concept check - Zero-shot: {has_zero}, Few-shot: {has_few}, Reasoning: {has_reasoning}"
    }


def eval_q10_cli_sequence(response: str) -> dict:
    """
    Q10: Podman & Ollama CLI sequence
    - Checks: 1) podman exec -it <container> bash
    - Checks: 2) ollama -h / ollama --help / ollama list
    - Checks: 3) ollama run granite4
    """
    resp_lower = sanitize_text(response).lower()
    has_exec = "podman exec" in resp_lower and "bash" in resp_lower
    has_ollama_help = "ollama" in resp_lower and ("-h" in resp_lower or "--help" in resp_lower or "list" in resp_lower)
    has_run_granite = "ollama run" in resp_lower and "granite" in resp_lower
    
    passed = has_exec and has_ollama_help and has_run_granite
    score = (0.33 if has_exec else 0) + (0.33 if has_ollama_help else 0) + (0.34 if has_run_granite else 0)
    
    return {
        "passed": passed,
        "score": round(score, 2),
        "feedback": f"Commands check - podman exec: {has_exec}, ollama help/list: {has_ollama_help}, ollama run granite: {has_run_granite}"
    }


def eval_q11_diagnostic_checklist(response: str) -> dict:
    """
    Q11: Environment diagnostic checklist
    - Checks 4 conditions: container running (`podman ps`), port mapping, model in list (`ollama list`), test run (`ollama run`)
    """
    resp_lower = sanitize_text(response).lower()
    has_ps = "podman ps" in resp_lower or "status up" in resp_lower or "running" in resp_lower
    has_port = "port" in resp_lower or "11434" in resp_lower or "mapped" in resp_lower
    has_list = "ollama list" in resp_lower or "granite" in resp_lower
    has_test_run = "ollama run" in resp_lower or "test prompt" in resp_lower or "inference" in resp_lower
    
    count = sum([has_ps, has_port, has_list, has_test_run])
    score = count / 4.0
    passed = (count >= 3)
    
    return {
        "passed": passed,
        "score": score,
        "feedback": f"Checklist items - Container up: {has_ps}, Port mapped: {has_port}, Model listed: {has_list}, Test run: {has_test_run}"
    }


def eval_q12_code_is_palindrome(response: str) -> dict:
    """
    Q12: Code Generation - is_palindrome(text: str) -> bool
    - Extracts python code
    - Checks docstring presence
    - Executes test cases in safe sandbox
    """
    match = re.search(r"```(?:python)?\n(.*?\n)```", sanitize_text(response), re.DOTALL)
    code = match.group(1) if match else response
    
    try:
        parsed = ast.parse(code)
        func_node = None
        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef) and node.name == "is_palindrome":
                func_node = node
                break
                
        if not func_node:
            return {"passed": False, "score": 0.0, "feedback": "Function `def is_palindrome` not found in response."}
            
        docstring = ast.get_docstring(func_node)
        has_docstring = bool(docstring and len(docstring.strip()) > 0)
        
        namespace = {}
        exec(compile(parsed, filename="<ast>", mode="exec"), namespace)
        fn = namespace.get("is_palindrome")
        
        test_results = [
            fn("racecar") == True,
            fn("radar") == True,
            fn("hello") == False,
            fn("A") == True,
            fn("level") == True
        ]
        
        all_tests_passed = all(test_results)
        score = (0.3 if has_docstring else 0.0) + (0.7 if all_tests_passed else 0.0)
        passed = has_docstring and all_tests_passed
        
        return {
            "passed": passed,
            "score": round(score, 2),
            "feedback": f"Docstring present: {has_docstring}, Test cases passed: {sum(test_results)}/{len(test_results)}"
        }
    except Exception as e:
        return {"passed": False, "score": 0.0, "feedback": f"Code parsing/execution error: {str(e)}"}

def extract_json_from_response(response: str) -> dict:
    """Helper to robustly extract and parse JSON from a response string."""
    # Try finding markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", sanitize_text(response), re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
    else:
        # Fallback: look for outermost brackets
        start = sanitize_text(response).find("{")
        end = sanitize_text(response).rfind("}")
        if start != -1 and end != -1 and start < end:
            json_str = sanitize_text(response)[start:end+1]
        else:
            json_str = sanitize_text(response)
    try:
        return json.loads(json_str)
    except Exception:
        return None


def eval_q13_rag_negative_constraint(response: str) -> dict:
    """Q13: Checks negative constraints and basic semantic concepts."""
    resp_lower = sanitize_text(response).lower().replace("-", "")
    
    # Negative check
    forbidden = ["vector", "database", "embeddings", "embedding"]
    found_forbidden = [w for w in forbidden if w in resp_lower]
    
    # Positive check (must mention at least 2 benefits of RAG/drawbacks of FT)
    benefits = ["cost", "train", "hallucinat", "up to date", "ground", "real time"]
    found_benefits = sum(1 for b in benefits if b in resp_lower)
    
    passed = (len(found_forbidden) == 0) and (found_benefits >= 2)
    score = 1.0 if passed else 0.0
    
    feedback = f"Forbidden words found: {found_forbidden} | Benefit concepts found: {found_benefits}/2"
    return {"passed": passed, "score": score, "feedback": feedback}


def eval_q14_temperature_parameter(response: str) -> dict:
    """Q14: Checks exact Ollama CLI command and temperature theory."""
    resp_lower = sanitize_text(response).lower()
    
    has_command = "/set parameter temperature 0" in resp_lower
    has_temp_0_concept = any(w in resp_lower for w in ["deterministic", "greedy", "exact", "consistent"])
    has_temp_high_concept = any(w in resp_lower for w in ["random", "creative", "diverse", "entropy"])
    
    passed = has_command and has_temp_0_concept and has_temp_high_concept
    
    feedback = f"Command: {has_command}, Temp 0: {has_temp_0_concept}, Temp 1.25: {has_temp_high_concept}"
    return {"passed": passed, "score": 1.0 if passed else 0.0, "feedback": feedback}


def eval_q15_rag_tabular_limitations(response: str) -> dict:
    """Q15: Parses JSON output for RAG limitations."""
    data = extract_json_from_response(sanitize_text(response))
    if not data:
        return {"passed": False, "score": 0.0, "feedback": "Failed to parse JSON."}
        
    has_tabular = "tabular_failure_reason" in data
    has_narrative = "summarization_success_reason" in data
    
    if has_tabular and has_narrative:
        val1 = str(data["tabular_failure_reason"]).lower()
        mentions_chunking = any(w in val1 for w in ["chunk", "unstructured", "text", "split"])
        passed = mentions_chunking
        return {"passed": passed, "score": 1.0 if passed else 0.5, "feedback": f"Chunking mentioned: {mentions_chunking}"}
        
    return {"passed": False, "score": 0.0, "feedback": "Missing required JSON keys."}


def eval_q16_agentic_sql_delegation(response: str) -> dict:
    """Q16: Parses JSON to verify agent/tool delegation for analytical tasks."""
    data = extract_json_from_response(sanitize_text(response))
    if not data:
        return {"passed": False, "score": 0.0, "feedback": "Failed to parse JSON."}
        
    has_pattern = "recommended_pattern" in data
    if has_pattern:
        val = str(data["recommended_pattern"]).lower()
        is_agentic = any(w in val for w in ["agent", "sql", "tool", "function", "api"])
        return {"passed": is_agentic, "score": 1.0 if is_agentic else 0.0, "feedback": f"Agentic keyword found: {is_agentic}"}
        
    return {"passed": False, "score": 0.0, "feedback": "Missing 'recommended_pattern' key."}


def eval_q17_function_calling_json(response: str) -> dict:
    """Q17: Checks for a strict function-calling JSON payload."""
    data = extract_json_from_response(sanitize_text(response))
    if not data:
        return {"passed": False, "score": 0.0, "feedback": "Failed to parse JSON."}
    
    # Function calls usually have 'name' and 'parameters' or 'arguments'
    resp_str = json.dumps(data).lower()
    has_tool_name = "web_search" in resp_str
    
    # FIXED: Allow either "tum" or "munich" (in case the model expands to Technical University of Munich)
    has_target_entity = "tum" in resp_str or "munich" in resp_str
    has_query = has_target_entity and "tuition" in resp_str and "2026" in resp_str
    
    passed = has_tool_name and has_query
    return {
        "passed": passed, 
        "score": 1.0 if passed else 0.0, 
        "feedback": f"Tool name: {has_tool_name}, Query args: {has_query}"
    }


def eval_q18_search_provider_tradeoffs(response: str) -> dict:
    """Q18: Evaluates JSON strictly mapping API requirements for DDG vs Tavily."""
    data = extract_json_from_response(sanitize_text(response))
    if not data:
        return {"passed": False, "score": 0.0, "feedback": "Failed to parse JSON."}
        
    try:
        ddg = data.get("duckduckgo", {})
        tav = data.get("tavily", {})
        
        # DDG does not need an API key, Tavily does.
        ddg_api = str(ddg.get("needs_api_key")).lower() in ["false", "no", "0"]
        tav_api = str(tav.get("needs_api_key")).lower() in ["true", "yes", "1"]
        
        passed = ddg_api and tav_api
        return {"passed": passed, "score": 1.0 if passed else 0.0, "feedback": f"DDG key check: {ddg_api}, Tavily key check: {tav_api}"}
    except Exception as e:
        return {"passed": False, "score": 0.0, "feedback": f"JSON Schema mismatch: {e}"}


def eval_q19_quantization_model_size(response: str) -> dict:
    """Q19: Evaluates JSON for size vs performance trade-offs."""
    data = extract_json_from_response(sanitize_text(response))
    if not data:
        return {"passed": False, "score": 0.0, "feedback": "Failed to parse JSON."}
        
    try:
        small = str(data.get("3B_model", {})).lower()
        large = str(data.get("15B_model", {})).lower()
        
        small_speed = any(w in small for w in ["fast", "latency", "speed", "quick", "memory"])
        large_logic = any(w in large for w in ["reasoning", "complex", "accurate", "context"])
        
        passed = small_speed and large_logic
        return {"passed": passed, "score": 1.0 if passed else 0.0, "feedback": f"Small=Speed: {small_speed}, Large=Logic: {large_logic}"}
    except Exception as e:
        return {"passed": False, "score": 0.0, "feedback": "JSON Schema mismatch."}


def eval_q20_zero_shot_context_clear(response: str) -> dict:
    """Q20: Checks definition and the specific Ollama /clear command."""
    resp_lower = sanitize_text(response).lower()
    
    has_def = any(w in resp_lower for w in ["without example", "no example", "no prior", "zero example"])
    has_command = "/clear" in resp_lower
    
    passed = has_def and has_command
    return {"passed": passed, "score": 1.0 if passed else 0.0, "feedback": f"Definition: {has_def}, Command (/clear): {has_command}"}


def eval_q21_search_agent_system_prompt(response: str) -> dict:
    """Q21: Checks system prompt for an exact forced refusal string."""
    exact_phrase = "I cannot find an answer based on trustworthy sources."
    passed = exact_phrase.lower() in sanitize_text(response).lower()
    return {"passed": passed, "score": 1.0 if passed else 0.0, "feedback": f"Found exact required refusal phrase: {passed}"}


def eval_q22_rag_verification_protocol(response: str) -> dict:
    """Q22: Validates JSON structure containing a 3-phase testing protocol."""
    data = extract_json_from_response(sanitize_text(response))
    if not data:
        return {"passed": False, "score": 0.0, "feedback": "Failed to parse JSON."}
    
    has_phases = all(k in data for k in ["phase_1", "phase_2", "phase_3"])
    if not has_phases:
         return {"passed": False, "score": 0.0, "feedback": "Missing phase_1, phase_2, or phase_3 keys."}
         
    p1 = str(data["phase_1"]).lower()
    p2 = str(data["phase_2"]).lower()
    p3 = str(data["phase_3"]).lower()
    
    # Check that the model actually generated substantive text for each phase (>30 chars)
    has_substantive_content = len(p1) > 30 and len(p2) > 30 and len(p3) > 30
    
    # FIXED: Instead of rigidly expecting specific words in specific phases, 
    # we verify the overall JSON object discusses core RAG evaluation concepts.
    combined_text = (p1 + " " + p2 + " " + p3)
    
    mentions_retrieval = any(w in combined_text for w in ["retrieval", "ingest", "document", "chunk", "context"])
    mentions_generation = any(w in combined_text for w in ["generation", "answer", "accuracy", "hallucination", "end-to-end"])
    mentions_evaluation = any(w in combined_text for w in ["evaluate", "verify", "test", "compare", "baseline", "metric", "robustness"])
    
    passed = has_substantive_content and mentions_retrieval and mentions_generation and mentions_evaluation
    
    score = 0.0
    if has_phases: score += 0.25
    if has_substantive_content: score += 0.25
    if (mentions_retrieval and mentions_generation and mentions_evaluation): score += 0.50
    
    return {
        "passed": passed, 
        "score": score, 
        "feedback": f"Has 3 phases: {has_phases}, Substantive content: {has_substantive_content}, Valid RAG concepts: {mentions_retrieval and mentions_generation and mentions_evaluation}"
    }

def eval_q23_prefill_decode_phases(response: str) -> dict:
    resp = sanitize_text(response).lower()
    has_prefill_compute = bool(re.search(r'prefill.*?compute\s*(?:-|)bound', resp))
    has_decode_memory = bool(re.search(r'decode.*?memory\s*(?:-|)bandwidth\s*(?:-|)bound', resp))
    has_ttft = bool(re.search(r'ttft.*?prefill|prefill.*?ttft', resp))
    has_tpot = bool(re.search(r'tpot.*?decode|decode.*?tpot', resp))
    score = sum([has_prefill_compute, has_decode_memory, has_ttft, has_tpot]) / 4.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q24_adaptation_ladder(response: str) -> dict:
    resp = sanitize_text(response).lower()
    req1_rag = bool(re.search(r'requirement 1:\s*rag', resp))
    req2_ft = bool(re.search(r'requirement 2:\s*fine\s*(?:-|)tuning', resp))
    score = sum([req1_rag, req2_ft]) / 2.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q25_vram_estimation(response: str) -> dict:
    # Expects output exactly like "Total VRAM: ~11 GB" or "Total VRAM: 11 GB"
    match = re.search(r'total vram:\s*~?\s*(\d+)\s*gb', sanitize_text(response), re.IGNORECASE)
    
    if not match:
        return {"passed": False, "score": 0.0, "feedback": "Did not find expected strictly formatted final answer."}
    
    vram_val = int(match.group(1))
    # Allow 10, 11, or 12 depending on exact math rounding the LLM used
    passed = vram_val in [10, 11, 12]
    
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "feedback": f"Extracted VRAM estimate: {vram_val} GB"
    }

def eval_q26_llamacpp_vs_vllm(response: str) -> dict:
    resp = sanitize_text(response).lower()
    
    llama_cpu = 'cpu offload' in resp or 'cpu' in resp
    llama_gguf = 'gguf' in resp
    vllm_paged = 'pagedattention' in resp or 'pagedkv' in resp or 'paged kv' in resp
    vllm_batch = 'continuous batching' in resp
    
    dev = re.search(r'llama\.cpp.*?(dev|single)', resp) is not None
    prod = re.search(r'vllm.*?(prod|multi)', resp) is not None

    checks = [llama_cpu or llama_gguf, vllm_paged, vllm_batch, dev, prod]
    score = sum(checks) / 5.0
    
    return {
        "passed": score >= 0.8, # Allow missing 1 minor detail
        "score": score,
        "feedback": f"llama(cpu/gguf):{checks[0]}, vllm(paged):{checks[1]}, vllm(batch):{checks[2]}, dev:{checks[3]}, prod:{checks[4]}"
    }

def eval_q27_mbu_performance(response: str) -> dict:
    resp = sanitize_text(response).lower()
    has_math = 'tpot' in resp and ('weight' in resp or 'kv' in resp)
    has_mem_bound = 'memory' in resp and 'bound' in resp
    has_saturation = 'saturat' in resp or '100%' in resp or 'hardware capacity' in resp
    
    score = sum([has_math, has_mem_bound, has_saturation]) / 3.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q28_evaluation_metrics(response: str) -> dict:
    resp = sanitize_text(response).lower()
    bleu = 'bleu: precision' in resp
    rouge = 'rouge: recall' in resp
    bert = 'bertscore: contextual embeddings' in resp
    perp = 'perplexity: model surprise' in resp
    
    score = sum([bleu, rouge, bert, perp]) / 4.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q29_stateful_vs_stateless(response: str) -> dict:
    resp = sanitize_text(response).lower()
    odoo_entity = 'entity' in resp
    odoo_wf = 'workflow' in resp
    odoo_hist = 'historical' in resp
    http_state = 'cookie' in resp or 'token' in resp or 'session' in resp
    
    score = sum([odoo_entity, odoo_wf, odoo_hist, http_state]) / 4.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q30_persistence_postgres(response: str) -> dict:
    resp = sanitize_text(response).lower()
    has_acid = 'acid' in resp
    has_json = 'json' in resp
    has_port = '5432' in resp
    has_wire = 'wire protocol' in resp
    
    score = sum([has_acid, has_json, has_port, has_wire]) / 4.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}


def eval_q31_reverse_proxy_nginx(response: str) -> dict:
    resp = sanitize_text(response).lower()
    has_ssl = 'ssl' in resp or 'tls' in resp
    has_443 = '443' in resp or 'https' in resp
    has_8069 = '8069' in resp
    # Regex explicitly supports "load balanc" or "load-balanc"
    has_lb = bool(re.search(r'load[\s\-]balanc', resp)) 
    
    score = sum([has_ssl, has_443, has_8069, has_lb]) / 4.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q32_business_crud_mapping(response: str) -> dict:
    resp = sanitize_text(response).upper()
    # Looking for '1: CREATE', '2: UPDATE, CREATE', etc.
    step1 = bool(re.search(r'1:\s*.*?CREATE', resp))
    step2 = bool(re.search(r'2:\s*.*?(UPDATE|CREATE)', resp))
    step3 = bool(re.search(r'3:\s*.*?UPDATE', resp))
    step4 = bool(re.search(r'4:\s*.*?(READ|SELECT|CREATE)', resp))
    
    score = sum([step1, step2, step3, step4]) / 4.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q33_docker_compose_schema(response: str) -> dict:
    response = sanitize_text(response) # CRITICAL: Removes \xa0 so YAML parses
    match = re.search(r"```(?:yaml)?\n(.*?\n)```", response, re.DOTALL | re.IGNORECASE)
    yaml_str = match.group(1) if match else response
    
    try:
        data = yaml.safe_load(yaml_str)
        services = data.get("services", {})
        
        odoo_svc, db_svc = None, None
        for k, v in services.items():
            if 'odoo' in v.get('image', ''): odoo_svc = v
            if 'postgres' in v.get('image', ''): db_svc = v
            
        if not odoo_svc or not db_svc:
            return {"passed": False, "score": 0.0, "feedback": "Missing odoo or db service declaration"}
            
        has_port = "8069:8069" in odoo_svc.get("ports", [])
        has_depends = "db" in str(odoo_svc.get("depends_on", "")) # Safely checks dict or list
        
        env_str = str(db_svc.get("environment", "")).upper()
        has_env = "POSTGRES_USER" in env_str and "POSTGRES_PASSWORD" in env_str
        has_vols = "volumes" in data
        
        score = sum([has_port, has_depends, has_env, has_vols]) / 4.0
        return {"passed": score == 1.0, "score": score, "feedback": ""}
        
    except Exception as e:
         return {"passed": False, "score": 0.0, "feedback": f"YAML Parse Error: {str(e)}"}


def eval_q34_gguf_file_naming(response: str) -> dict:
    resp = sanitize_text(response).lower()
    fam = 'granite' in resp
    ver = '3.1' in resp
    scale = '8b' in resp or '8 billion' in resp
    var = 'instruct' in resp
    prec = 'q5' in resp or '5-bit' in resp
    meth = 'k_l' in resp or 'k-quant' in resp
    
    score = sum([fam, ver, scale, var, prec, meth]) / 6.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q35_gguf_vs_safetensors(response: str) -> dict:
    resp = sanitize_text(response).lower()
    safe_mem = 'zero-copy' in resp or 'zero copy' in resp
    gguf_cpu = 'llama.cpp' in resp or 'cpu' in resp
    gguf_file = any(x in resp for x in ['single-file', 'unified', 'same file', 'bundles'])
    
    score = sum([safe_mem, gguf_cpu, gguf_file]) / 3.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}

def eval_q36_quantization_precision_scale(response: str) -> dict:
    resp = sanitize_text(response).lower()
    has_precs = all(x in resp for x in ['fp16', 'int8', 'q4', 'q2'])
    
    # Check if Q2/Q3 is explicitly associated with "severe" degradation
    severe_q2 = bool(re.search(r'q2.*?severe|severe.*?q2|q3.*?severe|2-bit.*?severe', resp))
    
    score = sum([has_precs, severe_q2]) / 2.0
    return {"passed": score == 1.0, "score": score, "feedback": ""}


# =========================================================================
# BENCHMARK SUITE WITH EXPECTED CRITERIA
# =========================================================================

TEST_SUITE = [
    {
        "id": 1,
        "cat": "Negative Constraint",
        "prompt": "Explain how to deploy Ollama using Podman. Use container flags like port mapping and volumes. Do not use the words 'docker', 'virtualization', or 'cloud'.",
        "expected": "Podman deployment instructions without using docker/virtualization/cloud.",
        "eval": lambda t: eval_q1_negative_constraint(t)["passed"],
    },
    {
        "id": 2,
        "cat": "Troubleshooting Command",
        "prompt": "I am getting a '/bin/bash^M: Defekter Interpreter' error in my script. Explain the cause and provide the exact command to fix it.",
        "expected": "Explanation of CRLF (Windows line endings) and sed -i 's/\\r$//' or dos2unix fix.",
        "eval": lambda t: eval_q2_troubleshooting_command(t)["passed"],
    },
    {
        "id": 3,
        "cat": "YAML Config",
        "prompt": "Create a VS Code Continue configuration for the 'granite4' model using the 'ollama' provider with the apiBase set to '[http://sose26dc.ucc.cit.tum.de:5002](http://sose26dc.ucc.cit.tum.de:5002)'. You MUST output it strictly as a valid YAML format (do NOT use JSON).",
        "expected": "Valid YAML containing the specified models list without using JSON braces.",
        "eval": lambda t: eval_q3_strict_yaml_config(t)["passed"],
    },
   {
        "id": 4,
        "cat": "Architecture Design",
        "prompt": "Explain a standard 3-tier application architecture. You must format your entire response as a single, valid JSON object with exactly three keys: 'tier_1', 'tier_2', and 'tier_3'. Provide the explanation for each tier as the string value for its respective key. Output ONLY JSON.",
        "expected": "Valid JSON object strictly containing the keys tier_1, tier_2, and tier_3.",
        "eval": lambda t: eval_q4_architecture_json(t)["passed"],
    },
    {
        "id": 5,
        "cat": "Code: is_prime",
        "prompt": "Write a python function `is_prime(n: int) -> bool` that checks if a number is prime. Include a docstring.",
        "expected": "Valid Python code that passes prime number unit tests and includes a docstring.",
        "eval": lambda t: eval_q5_code_is_prime(t)["passed"],
    },
    {
        "id": 6,
        "cat": "NL to SQL Flow",
        "prompt": "Explain the end-to-end data flow for a Natural Language to SQL database query system.",
        "expected": "Flow mentioning Prompt/LLM, SQL/DB generation, and a layered sequence.",
        "eval": lambda t: eval_q6_nl_to_sql_flow(t)["passed"],
    },
    {
        "id": 7,
        "cat": "Data Standardization",
        "prompt": "Outline a 4-step address standardization workflow.",
        "expected": "Workflow mentioning consent, iterating records, LLM formatting, and DB updating.",
        "eval": lambda t: eval_q7_data_standardization_workflow(t)["passed"],
    },
    {
        "id": 8,
        "cat": "Structured JSON",
        "prompt": "Generate a catchy slogan for a fictional smart coffee mug named 'ThermoBrew' priced at '$45.99'. You MUST output ONLY a valid JSON object with exactly three keys: 'product_name', 'price', and 'generated_slogan'. Do not include any conversational text.",
        "expected": "Valid JSON object strictly containing the keys product_name, price, and a non-empty generated_slogan.",
        "eval": lambda t: eval_q8_structured_json_slogan(t)["passed"],
    },
    {
        "id": 9,
        "cat": "Prompt Engineering",
        "prompt": "Compare Zero-Shot and Few-Shot Prompting for Coder-LLMs. Why does Few-Shot improve performance?",
        "expected": "Explanation of zero-shot vs few-shot and reasoning regarding formatting/patterns.",
        "eval": lambda t: eval_q9_prompt_engineering(t)["passed"],
    },
    {
        "id": 10,
        "cat": "CLI Sequence",
        "prompt": "What is the CLI sequence to execute bash in a podman container running ollama, check ollama help or list, and run the granite4 model?",
        "expected": "Commands containing podman exec bash, ollama list/help, and ollama run granite4.",
        "eval": lambda t: eval_q10_cli_sequence(t)["passed"],
    },
    {
        "id": 11,
        "cat": "Diagnostic Checklist",
        "prompt": "Provide an environment diagnostic checklist for verifying an Ollama deployment.",
        "expected": "Checklist mentioning podman ps, port mapping, ollama list, and ollama run.",
        "eval": lambda t: eval_q11_diagnostic_checklist(t)["passed"],
    },
    {
        "id": 12,
        "cat": "Code: is_palindrome",
        "prompt": "Write a python function `is_palindrome(text: str) -> bool` that checks if a string is a palindrome. Include a docstring.",
        "expected": "Valid Python code that passes palindrome unit tests and includes a docstring.",
        "eval": lambda t: eval_q12_code_is_palindrome(t)["passed"],
    },
    {
        "id": 13,
        "cat": "Negative Constraint",
        "prompt": "Explain why Retrieval-Augmented Generation (RAG) is preferred over fine-tuning for incorporating domain-specific enterprise knowledge into LLMs. You must discuss cost and model retraining. Do not use the words 'vector', 'database', or 'embeddings'.",
        "expected": "Mentions cost and training/hallucination while strictly obeying the negative constraint (0 occurrences of 'vector', 'database', or 'embeddings').",
        "eval": lambda t: eval_q13_rag_negative_constraint(t)["passed"],
    },
    {
        "id": 14,
        "cat": "LLM Inference Parameters",
        "prompt": "In an interactive Ollama CLI session, what exact command sets the model temperature parameter to zero? Explain the technical effect of setting temperature to 0 versus 1.25. State the exact command clearly.",
        "expected": "Identifies '/set parameter temperature 0' and explains deterministic greedy decoding vs random entropy.",
        "eval": lambda t: eval_q14_temperature_parameter(t)["passed"],
    },
    {
        "id": 15,
        "cat": "RAG limitations (JSON)",
        "prompt": "Explain why standard RAG pipelines fail at exact row counts on CSV files, but succeed at narrative summarization. Output ONLY a valid JSON object with two keys: 'tabular_failure_reason' and 'summarization_success_reason'.",
        "expected": "Valid JSON containing the specific keys and an explanation mentioning unstructured text/chunking.",
        "eval": lambda t: eval_q15_rag_tabular_limitations(t)["passed"],
    },
    {
        "id": 16,
        "cat": "Agentic Architecture (JSON)",
        "prompt": "According to LLMOps best practices, how should an enterprise AI system handle exact analytical SQL database queries? Output ONLY a valid JSON object with the keys 'recommended_pattern' (name the pattern/tool) and 'reasoning'.",
        "expected": "Valid JSON recommending an Agent, SQL tool, or function calling pattern.",
        "eval": lambda t: eval_q16_agentic_sql_delegation(t)["passed"],
    },
    {
        "id": 17,
        "cat": "Function Calling (JSON)",
        "prompt": "Generate the precise JSON tool call payload that an LLM should produce to query TUM international student tuition fees in 2026 using a tool named 'web_search'. Output ONLY valid JSON.",
        "expected": "Valid JSON tool call for 'web_search' with parameters for TUM, tuition, and 2026.",
        "eval": lambda t: eval_q17_function_calling_json(t)["passed"],
    },
    {
        "id": 18,
        "cat": "Search Providers (JSON)",
        "prompt": "Compare DuckDuckGo and Tavily Search as Agentic AI web providers. Output ONLY a valid JSON object formatted exactly like this: {\"duckduckgo\": {\"needs_api_key\": true/false, \"latency\": \"...\"}, \"tavily\": {\"needs_api_key\": true/false, \"latency\": \"...\"}}",
        "expected": "Valid JSON strictly mapping DuckDuckGo to needs_api_key=False, and Tavily to needs_api_key=True.",
        "eval": lambda t: eval_q18_search_provider_tradeoffs(t)["passed"],
    },
    {
        "id": 19,
        "cat": "Model Sizing (JSON)",
        "prompt": "Analyze the trade-offs between a 3B quantized model and a 15B model for coding assistants. Output ONLY a valid JSON object with the root keys '3B_model' and '15B_model'.",
        "expected": "Valid JSON associating 3B with speed/latency and 15B with logic/accuracy.",
        "eval": lambda t: eval_q19_quantization_model_size(t)["passed"],
    },
    {
        "id": 20,
        "cat": "Zero-Shot & Context",
        "prompt": "Define Zero-Shot Prompting. When benchmarking multiple prompts sequentially in an Ollama CLI session, what exact command must be executed to clear conversation memory?",
        "expected": "Defines zero-shot prompting (no examples) and specifies the '/clear' command.",
        "eval": lambda t: eval_q20_zero_shot_context_clear(t)["passed"],
    },
    {
        "id": 21,
        "cat": "Exact Match System Prompt",
        "prompt": "Write a system prompt for a web search agent enforcing trustworthy sources. You MUST include this exact fallback sentence verbatim: 'I cannot find an answer based on trustworthy sources.'",
        "expected": "System prompt containing the exact forced fallback string.",
        "eval": lambda t: eval_q21_search_agent_system_prompt(t)["passed"],
    },
    {
        "id": 22,
        "cat": "RAG Verification (JSON)",
        "prompt": "Describe a 3-phase testing protocol to verify an LLM's RAG augmentation. Output ONLY a valid JSON object with exactly three keys: 'phase_1', 'phase_2', and 'phase_3'.",
        "expected": "Valid JSON containing the three phases (baseline test, ingest document, verify knowledge).",
        "eval": lambda t: eval_q22_rag_verification_protocol(t)["passed"],
    },
    {
        "id": 23,
        "cat": "LLM Inference & Context Window Dynamics",
        "prompt": "Explain the two distinct computational phases of LLM inference: the Prefill phase and the Decode phase. Your response must explicitly state which phase is 'compute-bound' and which is 'memory-bandwidth bound'. Furthermore, explicitly state which phase primarily determines 'Time-To-First-Token (TTFT)' and which determines 'Time-Per-Output-Token (TPOT)'",
        "expected": "Correctly identifies Prefill as compute-bound (evaluates full prompt in one pass, determining TTFT and populating KV cache) and Decode as memory-bandwidth bound (generates one token at a time sequentially, reusing KV cache, determining TPOT) [5-7].",
        "eval": lambda t: eval_q23_prefill_decode_phases(t)["passed"],
    },
    {
        "id": 24,
        "cat": "Adaptation Ladder Selection",
        "prompt": "An enterprise needs to customize an LLM for: (1) Integrating frequently updating internal company policies that require precise source citations, and (2) Enforcing a strict, specialized JSON output schema and corporate tone. Based on the LLMOps Adaptation Ladder, state whether RAG or Fine-Tuning is appropriate for each. You must include the exact phrases 'Requirement 1: [Choice]' and 'Requirement 2: [Choice]' in your answer, followed by your justification.",
        "expected": "Correctly identifies RAG as appropriate for Requirement 1 (integrating updating policies) and Fine-Tuning as appropriate for Requirement 2 (enforcing JSON schema and tone) [8-10].",
        "eval": lambda t: eval_q24_adaptation_ladder(t)["passed"],
    },
    {
        "id": 25,
        "cat": "VRAM Sizing & Hardware Constraints",
        "prompt": "Calculate the estimated total VRAM (in GB) required to host an 8-billion parameter LLM quantized at 4-bit (Q4) precision with an 8,192 (8k) context window for 4 concurrent users. Include parameter weight memory, KV cache memory, and a standard 15% operational headroom. Conclude your response with the exact string: 'Total VRAM: ~[X] GB' where [X] is your final rounded integer estimate.",
        "expected": "Estimates parameter weight VRAM (~4.5 GB for 8B at Q4), KV cache memory based on context length x users (~5.2 GB), adds 15% headroom (~1.5 GB), totaling approximately 11 GB VRAM [12].",
        "eval": lambda t: eval_q25_vram_estimation(t)["passed"],
    },
    {
        "id": 26,
        "cat": "Inference Engines & Serving Frameworks",
        "prompt": "Compare llama.cpp and vLLM as local LLM inference engines. You must explicitly mention 'CPU offloading' or 'GGUF' for llama.cpp, and 'PagedAttention' (or PagedKV) and 'continuous batching' for vLLM. Conclude by stating which is best for single-user dev and which is best for multi-user production.",
        "expected": "Highlights that llama.cpp supports CPU layer offloading and GGUF for single-user/laptop environments [14, 17], whereas vLLM requires full model VRAM, utilizing PagedAttention / Paged KV cache and continuous batching for high-throughput multi-user production [14-16].",
        "eval": lambda t: eval_q26_llamacpp_vs_vllm(t)["passed"],
    },
    {
        "id": 27,
        "cat": "Performance Metrics (MBU & Memory Bandwidth)",
        "prompt": "Define Model Bandwidth Utilization (MBU) for LLM inference decoding. You must include the mathematical formula for MBU involving weights, KV cache, and TPOT. Explain why decoding at small batch sizes is 'memory-bandwidth bound', and what a near 100% MBU score indicates regarding the serving stack.",
        "expected": "Defines MBU = achieved memory bandwidth / peak memory bandwidth (where achieved = (weight bytes + KV cache bytes) / TPOT) [7]. Explains that small batch sizes reload all model weights per token, saturating GPU memory bandwidth, and near 100% MBU means the serving software fully saturates hardware capacity [18].",
        "eval": lambda t: eval_q27_mbu_performance(t)["passed"],
    },
    {
        "id": 28,
        "cat": "Offline Evaluation Metrics",
        "prompt": "Differentiate BLEU, ROUGE, BERTScore, and Perplexity. Explicitly map each metric to its primary focus by using these exact pairings in your text: 'BLEU: precision', 'ROUGE: recall', 'BERTScore: contextual embeddings', and 'Perplexity: model surprise'.",
        "expected": "Identifies BLEU (precision, n-gram overlap, translation) [19], ROUGE (recall, reference coverage, summarization) [19], BERTScore (embedding space similarity, paraphrasing) [19], and Perplexity (exp of mean log-likelihood, fluency without reference text) [20].",
        "eval": lambda t: eval_q28_evaluation_metrics(t)["passed"],
    },
    {
        "id": 29,
        "cat": "Stateful vs. Stateless Architecture",
        "prompt": "Compare stateful and stateless application architectures in enterprise system design [21-23]. Explain how an Enterprise Resource Planning (ERP) platform like Odoo acts as a stateful system across Entity, Workflow, and Historical dimensions [24, 25], and explain how a stateless protocol like HTTP can deliver a stateful user experience [26, 27].",
        "expected": "Defines stateful (retains memory/context across requests) vs stateless (each request processed in isolation) [22, 23]. Explains Odoo state dimensions (Entity: persistent customer records, Workflow: deal stage progression, Historical: audit logs) [25] and explains HTTP statefulness via client-side tokens/cookies or server-side session IDs + database lookups [27].",
        "eval": lambda t: eval_q29_stateful_vs_stateless(t)["passed"],
    },
    {
        "id": 30,
        "cat": "Durable Persistence & Postgres Wire Protocol",
        "prompt": "Why is application-only RAM storage insufficient for enterprise backend platforms [28]? Explain how PostgreSQL provides a persistent state layer, highlighting its ACID compliance, support for structured SQL and JSON [29, 30], and the function of the PostgreSQL Wire Protocol over port 5432 [31, 32].",
        "expected": "Notes RAM storage fails on server restart/crash [28]. Explains PostgreSQL persistence (disk storage, ACID compliance preventing transaction corruption, hybrid SQL/JSON) [29] and PostgreSQL Wire Protocol handling data formatting, authentication, and SSL security between application and DB over port 5432 [32].",
        "eval": lambda t: eval_q30_persistence_postgres(t)["passed"],
    },
    {
        "id": 31,
        "cat": "Reverse Proxy & Edge Security",
        "prompt": "An enterprise web application server operates on internal port 8069 [33]. Explain why exposing this application server directly to the public internet poses security and scalability risks [34], and detail three core operational functions performed by an NGINX reverse proxy positioned in front of it [34, 35].",
        "expected": "Identifies security/scalability risks of direct exposure [34]. Explains NGINX functions: SSL/TLS termination on HTTPS port 443 [34, 36], request shielding/routing to internal port 8069 [33, 34], and load balancing across backend instances [34, 35].",
        "eval": lambda t: eval_q31_reverse_proxy_nginx(t)["passed"],
    },
    {
        "id": 32,
        "cat": "Business Process to Database CRUD Mapping",
        "prompt": "In an enterprise workflow, trace the database operations for: (1) Create customer order, (2) Confirm order, (3) Process shipment, (4) Prepare invoice. Format your final answer as a strict list matching exactly: '1: [CRUD], 2: [CRUD], 3: [CRUD], 4: [CRUD]' using only the words CREATE, READ, UPDATE, or DELETE (multiple allowed per number).",
        "expected": "Maps: (1) Create order -> INSERT sales order (CREATE), (2) Confirm order -> UPDATE order status (UPDATE) & INSERT delivery record (CREATE), (3) Process shipment -> UPDATE delivery status & UPDATE inventory (UPDATE), (4) Prepare invoice -> SELECT order details (READ) & INSERT invoice (CREATE) [39].",
        "eval": lambda t: eval_q32_business_crud_mapping(t)["passed"],
    },
    {
        "id": 33,
        "cat": "Docker Compose Orchestration Blueprint",
        "prompt": "Write a complete `compose.yaml` configuration that orchestrates two connected services: an `odoo` web service running on image `odoo:18.0` exposed on port `8069` [40], and a `db` database service running `postgres:17` with environment variables `POSTGRES_USER=odoo`, `POSTGRES_PASSWORD=odoo`, and `POSTGRES_DB=postgres` [41]. Ensure strict container startup order (`depends_on`) [40] and named persistent volumes for both containers [40, 41].",
        "expected": "Provides valid YAML containing `services`, `web` with `image: odoo:18.0`, `ports: [\"8069:8069\"]`, `depends_on: [db]`, `db` with `image: postgres:17`, environment variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`), and declared volume mounts under top-level `volumes:` [40, 41].",
        "eval": lambda t: eval_q33_docker_compose_schema(t)["passed"],
    },
    {
        "id": 34,
        "cat": "GGUF Quantization File Naming Deconstruction",
        "prompt": "Deconstruct the GGUF file naming convention `granite-3.1-8b-instruct-Q5_K_L.gguf` [42, 43]. Identify the model family, version, parameter scale, fine-tuning variant, quantization precision bits, and quantization method type [42, 44-46].",
        "expected": "Deconstructs: Model Family = Granite, Version = 3.1, Parameter Scale = 8B (8 billion parameters), Variant = instruct (instruction-tuned vs chat), Quantization Precision = Q5 (5-bit quantization), Method = K_L (K-quant large variant) [42, 44-46].",
        "eval": lambda t: eval_q34_gguf_file_naming(t)["passed"],
    },
    {
        "id": 35,
        "cat": "GGUF vs. Safetensors Format Comparison",
        "prompt": "Compare GGUF and Safetensors model file formats for local LLM operations [46, 47]. Explain how Safetensors achieves true zero-copy loading and lower overhead [46], and explain why GGUF is preferred for CPU/GPU quantized inference frameworks like llama.cpp [17, 47].",
        "expected": "Explains Safetensors benefits (maps data directly to RAM/VRAM without intermediate copies, lazy loading, safe header parsing for PyTorch/HF) [46] vs GGUF benefits (unified single-file container holding quantized weights and metadata, engineered specifically for llama.cpp/Ollama CPU+GPU execution) [17, 47].",
        "eval": lambda t: eval_q35_gguf_vs_safetensors(t)["passed"],
    },
    {
        "id": 36,
        "cat": "Quantization Trade-offs & Precision Scale",
        "prompt": "Analyze the performance and VRAM trade-offs across weight precisions (FP16/BF16, INT8/Q8, Q4/Q5, and Q2/Q3) for a 7B/8B model. State the approximate VRAM required for each. You must explicitly identify which specific precision level (e.g., Q4 or Q2) represents the threshold where model quality degradation shifts to 'severe'.",
        "expected": "Outlines VRAM scale for ~7B model: FP16 (~14GB, reference quality), INT8/Q8 (~7-8GB, negligible loss), Q4/Q5 (~4-5GB, small/acceptable loss), Q2/Q3 (~2-3GB, severe quality loss). Identifies Q2/Q3 as the threshold where degradation becomes noticeable [47].",
        "eval": lambda t: eval_q36_quantization_precision_scale(t)["passed"],
    }
]

ttft_valid = []
valid_tps_list = []
total_tokens_generated = 0
total_generation_time = 0
passed_count = 0
csv_rows = []

print(f"Starting Performance Benchmark for: {MODEL}\n")

for item in TEST_SUITE:
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = ""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": item["prompt"]}],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=2500,
            temperature=0.1,
            # Bypasses reverse-proxy / gateway caches without altering prompt text
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                # Alternative parameter depending on server:
                # "thinking": {"type": "disabled"}
            },
            extra_headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "X-Request-Id": str(uuid.uuid4()),
            },
            # Bypasses per-user inference caching across OpenAI-compatible servers
            user=f"bench-{uuid.uuid4()}",
        )

        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta.content or ""
                if first_token_time is None and delta:
                    first_token_time = time.perf_counter()
                full_response += delta

            if getattr(chunk, "usage", None) and chunk.usage:
                token_count = chunk.usage.completion_tokens
            elif chunk.choices and chunk.choices[0].delta.content:
                token_count += 1

        end_time = time.perf_counter()

    except Exception as e:
        print(f"[{item['id']:02d}/36] {item['cat']:<36} | ERROR: {e}")
        csv_rows.append(
            {
                "Index": item["id"],
                "Category": item["cat"],
                "Status": "ERROR",
                "Expected Answer": item["expected"],
                "Raw Output": str(e),
            }
        )
        continue

    is_pass = item["eval"](full_response)
    if is_pass:
        passed_count += 1
        status = "PASS"
        # Passing items keep expected and output blank while leaving index visible
        csv_rows.append(
            {
                "Index": item["id"],
                "Category": item["cat"],
                "Status": "PASS",
                "Expected Answer": "",
                "Raw Output": "",
            }
        )
    else:
        status = "FAIL"
        # Failed items log the expected criteria and the full raw generation
        csv_rows.append(
            {
                "Index": item["id"],
                "Category": item["cat"],
                "Status": "FAIL",
                "Expected Answer": item["expected"],
                "Raw Output": full_response,
            }
        )

    if first_token_time:
        ttft = first_token_time - start_time
        gen_time = end_time - first_token_time
        ttft_valid.append(ttft)

        if token_count >= 15 and gen_time >= 0.1:
            tps = (token_count - 1) / gen_time
            valid_tps_list.append(tps)
            total_tokens_generated += token_count - 1
            total_generation_time += gen_time
            print(
                f"[{item['id']:02d}/36] {item['cat']:<36} | {status} | TTFT: {ttft:.3f}s | {tps:6.2f} tok/s ({token_count} tok)"
            )
        else:
            print(
                f"[{item['id']:02d}/36] {item['cat']:<36} | {status} | TTFT: {ttft:.3f}s | [Short Burst: {token_count} tok]"
            )
    else:
        print(
            f"[{item['id']:02d}/36] {item['cat']:<36} | {status} | TTFT: FAILED  | 0.00 tok/s"
        )

# =========================================================================
# WRITE CSV RESULTS
# =========================================================================

with open(
    CSV_OUTPUT_FILE, mode="w", newline="", encoding="utf-8"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "Index",
            "Category",
            "Status",
            "Expected Answer",
            "Raw Output",
        ],
    )
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nEvaluations exported to: {CSV_OUTPUT_FILE}")

# Aggregate calculations
aggregate_tps = (
    total_tokens_generated / total_generation_time
    if total_generation_time > 0
    else 0
)
median_ttft = statistics.median(ttft_valid) if ttft_valid else 0

print("\n" + "=" * 60)
print(f"STATS REPORT: {MODEL}")
print("=" * 60)
print(
    f"Accuracy Score:        {passed_count}/{len(TEST_SUITE)} ({passed_count/len(TEST_SUITE)*100:.1f}%)"
)
print(f"Median TTFT (P50):     {median_ttft:.3f}s")
print(f"Aggregate Decode Rate: {aggregate_tps:.2f} tok/s")
print("=" * 60)