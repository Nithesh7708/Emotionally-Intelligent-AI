from __future__ import annotations


def generate_response_text(emotion: str, transcript: str | None) -> str:
    cleaned = (transcript or "").strip()
    content = f'You said: "{cleaned}". ' if cleaned else ""

    templates: dict[str, str] = {
        "happy": (
            f"{content}I can hear the positive energy in your voice—love that! "
            "What’s making you feel so good right now?"
        ),
        "sad": (
            f"{content}I’m really sorry you’re feeling this way. "
            "That sounds heavy. Do you want to tell me what happened, or would you prefer a gentle reset together?"
        ),
        "angry": (
            f"{content}I hear the frustration. Let’s slow it down for a moment. "
            "What part is bothering you the most, and what would feel like a fair next step?"
        ),
        "fear": (
            f"{content}It makes sense to feel scared when things feel uncertain. "
            "You’re not alone here—what’s the biggest worry on your mind right now?"
        ),
        "neutral": (
            f"{content}I’m with you. What would you like to talk about or figure out today?"
        ),
    }

    return templates.get(emotion, templates["neutral"])

