"""Validate an explicit hard reasoning budget before native submission."""
from .errors import APIError


def prepare_thinking_budget(value, tokenizer, thinking, max_new_tokens):
    if value is None:
        value = 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise APIError(400, 'thinking_budget must be a non-negative uint32 integer')
    if not value:
        return 0, 0
    if not thinking or value + 1 >= max_new_tokens:
        raise APIError(400, 'thinking_budget requires thinking and room for a final answer')
    token = tokenizer.convert_tokens_to_ids('</think>')
    if isinstance(token, bool) or not isinstance(token, int) or not 0 < token < 2**32:
        raise APIError(400, 'thinking end token is unavailable')
    return value, token
