import csv
import statistics
import time
import uuid
from openai import OpenAI
from collections import defaultdict


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

def load_csv_context(file_paths):
    context_data = ""
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as file:
                context_data += f"\n--- Data from {path} ---\n"
                context_data += file.read() + "\n"
        except FileNotFoundError:
            print(f"Warning: Could not find {path}. Skipping.")
    return context_data

def eval_q1(response_text):
    text = response_text.lower()
    # Lowercase checks and simplified flexible price matching
    passed = "water bottle" in text and "19.99" in text
    return {"passed": passed}

def eval_q2(response_text):
    text = response_text.lower()
    passed = "road bike" in text and "1219.99" in text
    return {"passed": passed}

def eval_q3(response_text, sales_file="resources/sales.csv", products_file="resources/products.csv"):
    sales_counts = defaultdict(int)
    product_names = {}
    
    try:
        # Load product names
        with open(products_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [col.strip() for col in (reader.fieldnames or [])]
            for row in reader:
                product_names[row["product_id"].strip()] = row["product_name"].strip()

        # Sum counts from sales.csv
        with open(sales_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [col.strip() for col in (reader.fieldnames or [])]
            for row in reader:
                pid = row.get("product_id", "").strip()
                cnt = row.get("count", "0").strip()
                if pid and cnt.isdigit():
                    sales_counts[pid] += int(cnt)
                
        if not sales_counts:
            return {"passed": False, "error": "No sales data found"}

        # Identify the winner AND capture its count
        winner_id = max(sales_counts, key=sales_counts.get)
        winner_name = product_names.get(winner_id, winner_id)
        winner_count = sales_counts[winner_id]

        # Convert response to lowercase for checking
        text_lower = response_text.lower()
        
        # STRICT EVALUATION: Check for both name and the numeric count
        name_is_present = winner_name.lower() in text_lower
        count_is_present = str(winner_count) in text_lower
        
        is_pass = name_is_present and count_is_present

        # Update TEST_SUITE "Expected Answer" dynamically if it fails
        expected_str = f"{winner_name} with {winner_count} total sales"
        
        return {"passed": is_pass, "calculated_max_item": expected_str}

    except Exception as e:
        print(f"\n[eval_q3 Debug] Calculation failed with error: {e}")
        return {"passed": False}

def eval_q4(response_text):
    text = response_text.lower()
    passed = "p007" in text or "bike lock" in text and "5" in text
    return {"passed": passed}


BASE_URL = load_base_url()
API_KEY = load_api_key()
MODEL = "openai/gpt-oss-120b"          # openai/gpt-oss-120b || "ibm-granite/granite-4.2-8b" || "Qwen3/Qwen3" || "google/gemma-4-26B-A4B-it"
CSV_OUTPUT_FILE = "RAG_eval_results.csv"

# Provide the paths to your two CSV files here
CSV_FILES = ["resources/products.csv", "resources/sales.csv"] 
rag_context = load_csv_context(CSV_FILES)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

TEST_SUITE = [
    {
        "id": 1,
        "cat": "Least Expensive Product",
        "prompt": "What is the least expensive product?",
        "expected": "Water Bottle with a price of $19.99.",
        "eval": lambda t: eval_q1(t)["passed"],
    },
    {
        "id": 2,
        "cat": "Most Expensive Product",
        "prompt": "What is the most expensive product?",
        "expected": "Road Bike with a price of $1219.99.",
        "eval": lambda t: eval_q2(t)["passed"],
    },
    {
            "id": 3,
            "cat": "Most Sold Product",
            "prompt": "What is the most sold product and how many units were sold? Show your steps.",
            "expected": "Gloves (P004) with 444 total sales",
            "eval": lambda t: eval_q3(t)["passed"],
    },
    {
            "id": 4,
            "cat": "First Product Sold",
            "prompt": "What is the first sold product and how many units were sold?",
            "expected": "5 units of Bike Lock (P007) were sold first.",                
            "eval": lambda t: eval_q4(t)["passed"],
    },
]

total_tests = len(TEST_SUITE)
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
    
    system_prompt = (
        "You are a data assistant. Answer the user's question concisely based strictly "
        f"on the following CSV context:\n{rag_context}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]}
            ],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=2500,
            temperature=0.1,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True, "enable_rag": True},
            },
            extra_headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "X-Request-Id": str(uuid.uuid4()),
            },
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
        print(f"[{item['id']:02d}/{total_tests}] {item['cat']:<36} | ERROR: {e}")
        csv_rows.append({
            "Index": item["id"], "Category": item["cat"], "Status": "ERROR",
            "Expected Answer": item["expected"], "Raw Output": str(e),
        })
        continue

    is_pass = item["eval"](full_response)
    if is_pass:
        passed_count += 1
        status = "PASS"
    else:
        status = "FAIL"

    csv_rows.append({
        "Index": item["id"], "Category": item["cat"], "Status": status,
        "Expected Answer": item["expected"] if not is_pass else "",
        "Raw Output": full_response,
    })

    if first_token_time:
        ttft = first_token_time - start_time
        gen_time = end_time - first_token_time
        ttft_valid.append(ttft)

        if token_count >= 15 and gen_time >= 0.1:
            tps = (token_count - 1) / gen_time
            valid_tps_list.append(tps)
            total_tokens_generated += token_count - 1
            total_generation_time += gen_time
            print(f"[{item['id']:02d}/{total_tests}] {item['cat']:<36} | {status} | TTFT: {ttft:.3f}s | {tps:6.2f} tok/s ({token_count} tok)")
        else:
            print(f"[{item['id']:02d}/{total_tests}] {item['cat']:<36} | {status} | TTFT: {ttft:.3f}s | [Short Burst: {token_count} tok]")
    else:
        print(f"[{item['id']:02d}/{total_tests}] {item['cat']:<36} | {status} | TTFT: FAILED  | 0.00 tok/s")

try:
    with open(CSV_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Index", "Category", "Status", "Expected Answer", "Raw Output"])
        writer.writeheader()
        writer.writerows(csv_rows)
except IOError as e:
    print(f"\nFailed to save output to {CSV_OUTPUT_FILE}: {e}")

print("\n" + "="*55)
print(" BENCHMARK & RAG EVALUATION SUMMARY")
print("="*55)
print(f"Total Tests Run: {total_tests}")
print(f"Tests Passed:    {passed_count} ({passed_count/total_tests*100:.1f}%)")
if ttft_valid:
    print(f"Average TTFT:    {statistics.mean(ttft_valid):.3f}s")
if valid_tps_list:
    print(f"Average TPS:     {statistics.mean(valid_tps_list):.2f} tokens/s")
print(f"\nDetailed evaluation logs saved to: {CSV_OUTPUT_FILE}")

import csv
from collections import defaultdict

sales_file = "resources/sales.csv"
products_file = "resources/products.csv"

# 1. Load product names
product_names = {}
with open(products_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [col.strip() for col in (reader.fieldnames or [])]
    for row in reader:
        product_names[row["product_id"].strip()] = row["product_name"].strip()

# 2. Tally sales per product
sales_counts = defaultdict(int)
with open(sales_file, mode="r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [col.strip() for col in (reader.fieldnames or [])]
    for row in reader:
        pid = row.get("product_id", "").strip()
        cnt = row.get("count", "0").strip()
        if pid and cnt.isdigit():
            sales_counts[pid] += int(cnt)

# 3. Print the leaderboard
print("\n--- SALES LEADERBOARD ---")
for pid, total in sorted(sales_counts.items(), key=lambda x: x[1], reverse=True):
    name = product_names.get(pid, "Unknown Product")
    print(f"[{pid}] {name:<20}: {total} units sold")

# 4. Show the winner
winner_id = max(sales_counts, key=sales_counts.get)
winner_name = product_names.get(winner_id, "Unknown")
print("-" * 35)
print(f" WINNER: {winner_name} ({winner_id}) with {sales_counts[winner_id]} total sales\n")