"""
Text Utilities Module
Helper functions for text processing and Unicode handling
"""
import re


def decode_unicode_escapes(text: str) -> str:
    """
    Decode Unicode escape sequences in text (e.g., \\u2019 → ').

    This fixes AI-generated text that contains literal Unicode escape sequences
    like \\u2019 (right single quotation mark), \\u2026 (ellipsis), etc.

    Args:
        text: Text potentially containing Unicode escape sequences

    Returns:
        Text with Unicode escape sequences decoded to actual characters
    """
    if not text:
        return text

    def decode_match(match):
        """Decode a single Unicode escape sequence."""
        try:
            return match.group(0).encode('utf-8').decode('unicode-escape')
        except:
            # If decoding fails, return original
            return match.group(0)

    # Find and decode all \uXXXX patterns
    pattern = r'\\u[0-9a-fA-F]{4}'
    decoded = re.sub(pattern, decode_match, text)

    return decoded


def clean_ai_text(text: str) -> str:
    """
    Clean AI-generated text by fixing common encoding issues.

    Args:
        text: AI-generated text

    Returns:
        Cleaned text with proper Unicode characters
    """
    if not text:
        return text

    # Decode Unicode escape sequences
    text = decode_unicode_escapes(text)

    # Additional cleanup if needed
    # (add more cleaning rules here in the future if needed)

    return text
