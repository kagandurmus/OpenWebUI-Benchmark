import time
import json
import re
import uuid
import statistics
from openai import OpenAI

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
MODEL = "MODEL_NAME"  # for example "Qwen/Qwen3-32B" or openai/gpt-oss-120b

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# =========================================================================
# PROGRAMMATIC EVALUATORS (ROBUST STRING & REGEX MATCHING)
# =========================================================================

def eval_negative_constraint(text):
    # Uses word boundaries (\b) to prevent failing on words like "microservices"
    # Matches "cloud", "clouds", "software", "service", "services"
    forbidden = re.compile(r'\b(cloud|software|service)s?\b', re.IGNORECASE)
    return not forbidden.search(text) and "watsonx" in text.lower()

def eval_json_schema(text):
    # Isolates the JSON block even if wrapped in markdown or conversational filler
    block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    json_str = block_match.group(1) if block_match else text

    # Strip everything before the first '{' and after the last '}' (non-greedy fallback)
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return False
        
    json_str = json_str[start_idx:end_idx+1]
    
    try:
        data = json.loads(json_str)
        return {"course_name", "target_audience", "duration_weeks"}.issubset(data.keys())
    except:
        return False

def eval_fact_granite(text):
    # Word boundaries prevent matching "1116" or "12th"
    has_tokens = bool(re.search(r'\b12\s*(?:trillion|t)\b', text, re.IGNORECASE))
    has_langs = bool(re.search(r'\b116\b', text))
    return has_tokens and has_langs

def eval_hardware_qa(text):
    # \b ensures we match the acronym "MMA" and not the string inside "comma" or "command"
    return bool(re.search(r'\bmma\b', text, re.IGNORECASE) or "matrix math accelerator" in text.lower())

def eval_langchain(text):
    # Exact parameter isolation so '50' does not trigger a false positive on '500'
    has_500 = bool(re.search(r'\b500\b', text))
    has_50 = bool(re.search(r'\b50\b', text))
    has_logic = bool(re.search(r'(textsplitter|splitt)', text, re.IGNORECASE))
    return has_500 and has_50 and has_logic

def eval_vram(text):
    # Ensures '32' is tied to 'GB' so echoing the prompt ("a 32-billion parameter model") fails
    return bool(re.search(r'\b32\s*(?:gb|gigabytes?|g)\b', text, re.IGNORECASE))

def eval_chip_math(text):
    return bool(re.search(r'\b30\.72\b', text) or re.search(r'\b30,?720\b', text))

def eval_metrics(text):
    # Isolates exact metric values (e.g., 0.9, 0.90, 90%, 0.818, 0.82)
    has_precision = bool(re.search(r'\b(0\.90?|90(?:\.0)?\s*%)\b', text))
    has_recall = bool(re.search(r'\b(0\.818|0\.82|81\.8\s*%)\b', text))
    return has_precision and has_recall

def eval_rag_rejection(text):
    # Fails instantly if the model pulls 1991 (Python's real release date) from parametric memory
    has_hallucinated_date = bool(re.search(r'\b(1991|91|1989)\b', text))
    signals = ["not mentioned", "not provided", "does not state", "cannot answer", "unknown", "no information", "does not contain"]
    has_rejection = any(sig in text.lower() for sig in signals)
    return has_rejection and not has_hallucinated_date

def eval_search_formulation(text):
    # Enforces uppercase boolean operators as requested in the prompt constraint
    return bool(re.search(r'\b(AND|OR)\b', text))

def eval_planning(text):
    # Ensures steps 1 through 4 exist independently (so '14' doesn't trigger 1 and 4)
    has_numbers = all(re.search(rf'\b{i}\b', text) for i in ["1", "2", "3", "4"])
    return has_numbers and any(w in text.lower() for w in ["step", "phase", "migration", "plan"])


# =========================================================================
# CORRECTED BENCHMARK SUITE
# =========================================================================

TEST_SUITE = [
    # Strict Instruction Following
    {"id": 1, "cat": "Negative Constraint", "prompt": "Explain watsonx value regarding data sovereignty. Do not use the words 'cloud', 'software', or 'service'.", "eval": eval_negative_constraint},
    {"id": 2, "cat": "Strict JSON Schema", "prompt": "Output a valid JSON object with keys: 'course_name', 'target_audience', 'duration_weeks' for an AI ethics course. Output ONLY JSON.", "eval": eval_json_schema},
    # Q&A capabilities
    {"id": 3, "cat": "Factual Q&A", "prompt": "What is the training scale and language coverage of Granite 3.0 8B?", "eval": eval_fact_granite},
    {"id": 4, "cat": "Hardware Q&A", "prompt": "Explain the AI acceleration hardware on IBM Power10 cores.", "eval": eval_hardware_qa},
    {"id": 5, "cat": "AI Guardrails", "prompt": "What is the primary role of IBM Granite Guardian models?", "eval": lambda t: any(k in t.lower() for k in ["guardrail", "safety", "hallucination", "risk", "toxicity"])},
    # Coding taks
    {"id": 6, "cat": "LangChain Splitter", "prompt": "Write a Python snippet using langchain to split text with chunk_size 500 and overlap 50.", "eval": eval_langchain},
    {"id": 7, "cat": "PyTorch Arch", "prompt": "Write a PyTorch nn.Module with two linear layers and ReLU activation.", "eval": lambda t: "nn.linear" in t.lower() and "relu" in t.lower()},
    {"id": 8, "cat": "REST API Request", "prompt": "Write a Python snippet using `requests` to send a POST request with headers and a JSON body.", "eval": lambda t: "requests.post" in t and "headers=" in t and "json=" in t},
    # Math and reasoning
    {"id": 9, "cat": "VRAM Calculation", "prompt": "What is the exact base VRAM in GB required to hold a 32-billion parameter model in INT8 precision?", "eval": eval_vram},
    {"id": 10, "cat": "Chip Operations", "prompt": "Calculate theoretical ops/sec for a 15-core chip at 4.0 GHz doing 512 ops/cycle per core. Show final number in Tera-ops.", "eval": eval_chip_math},
    {"id": 11, "cat": "Metric Arithmetic", "prompt": "Given TP=45, FP=5, FN=10, compute exact Precision and Recall.", "eval": eval_metrics},
    # RAG and retrieval: adhering to context, not hallucinating
    {"id": 12, "cat": "RAG Needle", "prompt": "Context: 'Project status: ALFA-77 is deployed on Power10.' Question: What is the project code?", "eval": lambda t: "alfa-77" in t.lower()},
    {"id": 13, "cat": "RAG Synthesis", "prompt": "Context A: 'Granite 8B is for servers.' Context B: 'Granite MoE 3B is for mobile edge.' Question: Which model is suited for edge deployment?", "eval": lambda t: bool(re.search(r'\b(moe 3b|3b)\b', t.lower()))},
    {"id": 14, "cat": "RAG Rejection", "prompt": "Context: 'watsonx.data is a lakehouse.' Question: What year was Python invented? Answer strictly from context.", "eval": eval_rag_rejection},
    # Tooling and search
    {"id": 15, "cat": "Tool Schema", "prompt": "Format a tool call to `query_course_db` with `topic`='Generative AI' and `level`='Beginner'.", "eval": lambda t: "query_course_db" in t and "Generative AI" in t and "Beginner" in t},
    {"id": 16, "cat": "Tool Routing", "prompt": "You have tools: `search_web` and `calculate_math`. To find the square root of Granite 3.0's parameter count, which tool must you execute first?", "eval": lambda t: "search_web" in t.lower() or "search" in t.lower()},
    # Multi-shot and few-shot reasoning
    {"id": 17, "cat": "Multi-Shot Sentiment", "prompt": "Text: 'Course was bad' Sentiment: Negative\nText: 'Great lab' Sentiment: Positive\nText: 'Documentation was clear' Sentiment:", "eval": lambda t: "positive" in t.lower() and "false positive" not in t.lower()},
    {"id": 18, "cat": "Multi-Shot Code", "prompt": "Scikit: `LogisticRegression(max_iter=100)` -> Snap ML: `SnapLogisticRegression(max_iter=100)`\nScikit: `RandomForestClassifier(n_estimators=50)` -> Snap ML:", "eval": lambda t: "snaprandomforestclassifier" in t.lower()},
    # Search and planning
    {"id": 19, "cat": "Search Formulation", "prompt": "Write an advanced search query using boolean operators to find Granite 3.0 speculative decoding.", "eval": eval_search_formulation},
    {"id": 20, "cat": "Planning", "prompt": "Outline a numbered 4-step migration plan to IBM watsonx.", "eval": eval_planning}
]


ttft_valid = []
valid_tps_list = []
total_tokens_generated = 0
total_generation_time = 0
passed_count = 0

print(f"Starting Performance Benchmark for: {MODEL}\n")

for item in TEST_SUITE:
    salted_prompt = f"[Run-ID: {uuid.uuid4()}]\n{item['prompt']}"
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = ""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": salted_prompt}],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=2500, # This prevents the model from generating excessively long outputs that could skew the timing metrics
            temperature=0.1
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
        print(f"[{item['id']:02d}/20] {item['cat']:<22} | ERROR: {e}")
        continue

    is_pass = item["eval"](full_response)
    if is_pass: passed_count += 1
    status = "PASS" if is_pass else "FAIL"

    if first_token_time:
        ttft = first_token_time - start_time
        gen_time = end_time - first_token_time
        ttft_valid.append(ttft)

        # Discard burst artifacts: only calculate discrete tok/s if output >= 15 tokens and gen_time >= 0.1s
        if token_count >= 15 and gen_time >= 0.1:
            tps = (token_count - 1) / gen_time
            valid_tps_list.append(tps)
            total_tokens_generated += (token_count - 1)
            total_generation_time += gen_time
            print(f"[{item['id']:02d}/20] {item['cat']:<22} | {status} | TTFT: {ttft:.3f}s | {tps:6.2f} tok/s ({token_count} tok)")
        else:
            print(f"[{item['id']:02d}/20] {item['cat']:<22} | {status} | TTFT: {ttft:.3f}s | [Short Burst: {token_count} tok]")
    else:
        print(f"[{item['id']:02d}/20] {item['cat']:<22} | {status} | TTFT: FAILED  | 0.00 tok/s")

# Aggregate calculations
aggregate_tps = total_tokens_generated / total_generation_time if total_generation_time > 0 else 0
median_ttft = statistics.median(ttft_valid) if ttft_valid else 0

print("\n" + "="*60)
print(f"STATS REPORT: {MODEL}")
print("="*60)
print(f"Accuracy Score:        {passed_count}/{len(TEST_SUITE)} ({passed_count/len(TEST_SUITE)*100:.1f}%)")
print(f"Median TTFT (P50):     {median_ttft:.3f}s")
print(f"Aggregate Decode Rate: {aggregate_tps:.2f} tok/s")
print("="*60)