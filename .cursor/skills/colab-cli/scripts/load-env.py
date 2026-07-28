"""Load /content/.env into the persistent Colab kernel, then delete it."""

from __future__ import annotations

import os
import re
from pathlib import Path


ENV_PATH = Path("/content/.env")
KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def decode_double_quoted(value: str) -> str:
    replacements = {
        "\\": "\\",
        '"': '"',
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    decoded: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "\\" or index + 1 == len(value):
            decoded.append(character)
            index += 1
            continue

        escaped = value[index + 1]
        replacement = replacements.get(escaped)
        if replacement is None:
            decoded.extend(("\\", escaped))
        else:
            decoded.append(replacement)
        index += 2
    return "".join(decoded)


def parse_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        escaped = False
        closing_index = None
        for index, character in enumerate(value[1:], start=1):
            if character == quote and not escaped:
                closing_index = index
                break
            escaped = character == "\\" and not escaped
            if character != "\\":
                escaped = False
        if closing_index is None:
            raise ValueError("Unclosed quoted value")
        trailing = value[closing_index + 1 :].strip()
        if trailing and not trailing.startswith("#"):
            raise ValueError("Unexpected text after quoted value")
        content = value[1:closing_index]
        if quote == '"':
            content = decode_double_quoted(content)
        return content
    # In unquoted values, a comment starts only at whitespace followed by '#'.
    value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    return value


if not ENV_PATH.is_file():
    raise FileNotFoundError(f"Expected uploaded environment file at {ENV_PATH}")

try:
    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        ENV_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()

        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not KEY_RE.fullmatch(key):
            raise ValueError(f"Invalid .env entry on line {line_number}")
        try:
            value = parse_value(raw_value)
        except ValueError as error:
            raise ValueError(f"Invalid .env value on line {line_number}") from error
        if "\x00" in value:
            raise ValueError(f"NUL byte in .env value on line {line_number}")
        parsed[key] = value

    # Apply only after every entry has been parsed and validated successfully.
    os.environ.update(parsed)
finally:
    ENV_PATH.unlink(missing_ok=True)

print(f"Loaded {len(parsed)} environment variable(s); deleted {ENV_PATH}")
