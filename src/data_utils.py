from typing import List, Dict, Optional

def parse_answer_logprobs(response: ChatCompletion) -> List[float]:
    prompt_logprobs: List[Dict] = response.prompt_logprobs[::-1]

    # Hacky: this only works for llama-3 models
    assert '128006' in prompt_logprobs[3], f"Invalid prompt logprobs: {prompt_logprobs}"
    prompt_logprobs = prompt_logprobs[4:] # Skip the added generation prompt

    answer_logprobs: List[float] = []
    for logprobs in prompt_logprobs:
        logprobs: Dict[str, Dict]
        prob_infos: List[Dict] = sorted(list(logprobs.values()), key=lambda x: x['rank'])
        if prob_infos[-1]['decoded_token'] == '<|end_header_id|>':
            # also skip the "\n\n" token
            answer_logprobs = answer_logprobs[:-1]
            break

        prob = prob_infos[-1]['logprob']
        answer_logprobs.append(prob)

    return answer_logprobs