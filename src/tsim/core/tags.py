"""Helpers for tags that encode tsim-specific gate metadata."""

T_TAG = "T"
_T_USER_TAG_PREFIX = f"{T_TAG}:"


def encode_t_tag(user_tag: str = "") -> str:
    """Encode a T-family gate tag while preserving an optional user tag."""
    if not user_tag:
        return T_TAG
    return f"{_T_USER_TAG_PREFIX}{user_tag}"


def is_t_tag(tag: str) -> bool:
    """Return whether a Stim tag encodes a tsim T-family gate."""
    return tag == T_TAG or tag.startswith(_T_USER_TAG_PREFIX)


def decode_t_user_tag(tag: str) -> str:
    """Return the user tag attached to an encoded T-family gate tag."""
    if tag == T_TAG:
        return ""
    if tag.startswith(_T_USER_TAG_PREFIX):
        return tag[len(_T_USER_TAG_PREFIX) :]
    raise ValueError(f"Tag does not encode a T-family gate: {tag!r}")
