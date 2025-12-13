"""
Text processing and LLM utilities for podcast pipeline.
Includes Korean transliteration, speaker tagging, and LLM parsing functions.
"""

import re
import json
import ast
from typing import List
from g2pk import G2p

# English pattern for Korean transliteration
ENG_PATTERN = re.compile(r"[A-Za-z]+")

# Cost calculation constants (for LLM usage tracking)
COST_PER_MILLION_INPUT = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4-turbo": 10.00,
    "gpt-3.5-turbo": 0.50,
}

COST_PER_MILLION_OUTPUT = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-4-turbo": 30.00,
    "gpt-3.5-turbo": 1.50,
}


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of API usage based on model name and token counts.

    Args:
        model_name: Name of the LLM model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Total cost in USD
    """
    input_cost_per_million = COST_PER_MILLION_INPUT.get(model_name, 0.0)
    output_cost_per_million = COST_PER_MILLION_OUTPUT.get(model_name, 0.0)

    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million

    total_cost = input_cost + output_cost
    return total_cost


def speaker_tagged_text(data):
    """
    Generate speaker-tagged text from segment data.

    Args:
        data: List of segments with 'speaker' and 'text' fields

    Returns:
        String with speaker tags and text
    """
    result = []
    for item in data:
        speaker = item.get("speaker", "Unknown")
        text = item.get("text", "")
        result.append(f"[{speaker}]: {text}")
    return "\n".join(result)


def parse_speaker_summary(llm_output: str) -> list | None:
    """
    LLM이 출력한 문자열에서 JSON 배열을 추출하고 파싱합니다.
    'json' 접두사, 코드 블록(```), 앞뒤 공백 등을 처리합니다.
    """
    if not llm_output:
        return None

    try:
        # ```json ... ``` 또는 ``` ... ``` 같은 코드 블록 제거
        # 정규 표현식을 사용하여 대괄호 '[' 와 ']' 사이의 내용을 찾음
        match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            # JSON 문자열을 파이썬 객체로 변환 (list of dicts)
            return json.loads(json_str)
        else:
            print("Parsing Error: 유효한 JSON 배열 형식([])을 찾을 수 없습니다.")
            return None

    except json.JSONDecodeError as e:
        print(f"JSON 파싱 에러: {e}")
        return None
    except Exception as e:
        print(f"알 수 없는 파싱 에러: {e}")
        return None


def process_llm_diarization_output(llm_output: str) -> list[dict]:
    """
    Process LLM output for diarization results.

    Args:
        llm_output: Raw LLM output string

    Returns:
        List of dictionaries containing diarization data
    """
    # 1. LLM 출력에서 ```json ... ``` 코드 블록 찾기
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_output)
    if not json_match:
        # 만약 ```json 블록이 없다면, 문자열 전체를 파싱 시도
        json_string = llm_output
    else:
        json_string = json_match.group(1)

    # 2. JSON 문자열을 파이썬 객체로 파싱
    try:
        llm_data = json.loads(json_string)
    except json.JSONDecodeError:
        # LLM이 Python 리스트 형식('[{"text":...}]')으로 출력했을 경우를 대비
        try:
            # ast.literal_eval은 보안에 더 안전한 eval 버전입니다.
            llm_data = ast.literal_eval(json_string)
        except (ValueError, SyntaxError) as e:
            print(f"오류: JSON 및 Python 리터럴 파싱에 모두 실패했습니다. {e!r}")
            return []

    return llm_data


def ko_transliterate_english(text: str) -> str:
    """
    입력 문자열에서 영어 구간만 찾아 한글 발음으로 변환합니다.

    Args:
        text: Input text with English words

    Returns:
        Text with English transliterated to Korean pronunciation
    """
    def _repl(m: re.Match) -> str:
        segment = m.group(0)
        return G2p(segment)
    return ENG_PATTERN.sub(_repl, text)


def ko_process_json(input_list: List[dict]) -> None:
    """
    Process JSON list to transliterate English to Korean.

    Args:
        input_list: List of dictionaries with 'text' field
    """
    for entry in input_list:
        text = entry.get("text", "")
        # 영어 포함 시 변환
        if re.search(r"[A-Za-z]", text):
            entry["text"] = ko_transliterate_english(text)
