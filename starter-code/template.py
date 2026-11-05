"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
# import os
# import time
# from openai import OpenAI

# def call_openai(
#     prompt: str,
#     model: str = OPENAI_MODEL,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     max_tokens: int = 256,
# ) -> tuple[str, float, dict]:
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     start = time.time()
#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=temperature,
#         top_p=top_p,
#         max_tokens=max_tokens,
#     )
#     latency = time.time() - start

#     text = response.choices[0].message.content or ""
#     usage = {
#         "input_tokens": response.usage.prompt_tokens,
#         "output_tokens": response.usage.completion_tokens,
#     }
#     return text, latency, usage

# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # đặt ở đầu file, ngoài hàm

def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    # 1. Khởi tạo client, key lấy từ .env
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 2. Cấu hình tham số sinh văn bản
    config = types.GenerateContentConfig(
    temperature=temperature,
    top_p=top_p,
    max_output_tokens=max_tokens,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)

    # 3. Gọi API và đo latency
    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start

    # 4. Lấy text và usage
    text = response.text or ""
    meta = response.usage_metadata
    usage = {
        "input_tokens": meta.prompt_token_count or 0,
        "output_tokens": meta.candidates_token_count or 0,
    }

    return text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
# def call_anthropic(
#     prompt: str,
#     model: str = ANTHROPIC_MODEL,
#     temperature: float = 0.7,
#     top_p: float = 0.9,
#     max_tokens: int = 256,
# ) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    #TODo: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
    # raise NotImplementedError("Implement call_anthropic")


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
# Ba model Gemini dùng để so sánh: {tên key trong kết quả: tên model}
GEMINI_COMPARE_MODELS = {
    "gemini_flash": "gemini-3.5-flash",
    "gemini_flash_lite": "gemini-3.8-flash",
    "gemini_flash_lite_31": "gemini-3.1-flash-lite",
}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost = (input * input_rate + output * output_rate) / 1,000,000."""
    rates = PRICING_1M_TOKENS.get(model)
    if rates is None:
        return 0.0  # model chưa có trong bảng giá
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def compare_models(prompt: str) -> dict:
    """Gọi 3 model Gemini với cùng một prompt và trả về bảng so sánh."""
    results = {}
    for key, model_name in GEMINI_COMPARE_MODELS.items():
        try:
            text, latency, usage = retry_with_backoff(
            lambda: call_gemini(prompt, model=model_name),
            max_retries=3,
            base_delay=2.0,
        )
            results[key] = {
                "response": text,
                "latency": latency,
                "cost": _calc_cost(
                    model_name, usage["input_tokens"], usage["output_tokens"]
                ),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
            }
        except Exception as e:
            # Một model lỗi (404/503...) không làm hỏng cả phép so sánh
            results[key] = {
                "response": f"[ERROR] {e}",
                "latency": 0.0,
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.
 
    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=1024,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
 
    max_turns = 3
    history: list[types.Content] = []  # mỗi turn = 1 user + 1 model
 
    while True:
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break
 
        if user_input.lower() in {"quit", "exit"}:
            print("Tạm biệt!")
            break
        if not user_input:
            continue
 
        user_msg = types.Content(role="user", parts=[types.Part(text=user_input)])
 
        try:
            print("Bot: ", end="", flush=True)
            reply_parts: list[str] = []
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=history + [user_msg],
                config=config,
            )
            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    reply_parts.append(chunk.text)
            print()
        except Exception as e:
            # Lỗi thì không lưu turn này vào history
            print(f"\n[ERROR] {e}")
            continue
 
        reply = "".join(reply_parts)
        model_msg = types.Content(role="model", parts=[types.Part(text=reply)])
        history.extend([user_msg, model_msg])
        history = history[-max_turns * 2 :]  # chỉ giữ 3 turn gần nhất


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).
 
    Returns:
        The return value of fn() on success.
 
    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(base_delay * (2**attempt))
    raise last_exc  # type: ignore[misc]
# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.
 
    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    output: list[dict] = []
    for prompt in prompts:
        result = dict(compare_models(prompt))
        result["prompt"] = prompt
        output.append(result)
    return output

# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def _md_cell(text: str, limit: int | None = None) -> str:
    """Làm sạch nội dung ô Markdown: bỏ xuống dòng, escape dấu |, cắt ngắn nếu cần."""
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    if limit is not None and len(text) > limit:
        text = text[:limit] + "..."
    return text.replace("|", "\\|")
 
 
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.
 
    Columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
 
    for item in results:
        prompt = _md_cell(item.get("prompt", ""))
        for model_key, stats in item.items():
            if model_key == "prompt" or not isinstance(stats, dict):
                continue
            response = _md_cell(stats.get("response", ""), limit=50)
            latency = f"{stats.get('latency', 0.0):.2f}s"
            tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
            cost = f"${stats.get('cost', 0.0):.6f}"
            lines.append(
                f"| {prompt} | {model_key} | {response} | {latency} | {tokens} | {cost} |"
            )
 
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
