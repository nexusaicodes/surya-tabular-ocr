import re
from typing import List, Dict

from surya.recognition.schema import TextChar


def extract_tags(proposed_tags: List[str]) -> List[str]:
    tags = []
    for tag in proposed_tags:
        tag_match = re.match(tag_pattern, tag)
        if not tag_match:
            continue

        if not tag_match.group(1) == "/":
            continue

        tags.append(tag_match.group(2))
    return tags


tag_pattern = re.compile(r"<(/?)([a-z]+)([^>]*)>?", re.IGNORECASE)


def fix_unbalanced_tags(
    text_chars: List[TextChar], special_tokens: Dict[str, list]
) -> List[TextChar]:
    self_closing_tags = ["br"]

    open_tags = []

    format_tags = extract_tags(special_tokens["formatting"])

    for char in text_chars:
        if len(char.text) <= 1:
            continue

        tag_match = re.match(tag_pattern, char.text)
        if not tag_match:
            continue

        is_closing = tag_match.group(1) == "/"
        tag_name = tag_match.group(2).lower()

        if tag_name not in format_tags:
            continue

        if tag_name in self_closing_tags:
            continue

        # Self-closing tags
        if tag_match.group(3) and tag_match.group(3).strip().endswith("/"):
            continue

        if is_closing:
            if open_tags and open_tags[-1] == tag_name:
                open_tags.pop()
        else:
            open_tags.append(tag_name)

    for tag in open_tags:
        text_chars.append(
            TextChar(
                text=f"</{tag}>",
                confidence=0,
                polygon=[[0, 0], [1, 0], [1, 1], [0, 1]],
                bbox_valid=False,
            )
        )
    return text_chars
