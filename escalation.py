def check_escalation(message, intent, historical_reply):

    message_lower = message.lower()

    # Sensitive issues
    sensitive_words = [
        "hacked",
        "stolen",
        "fraud",
        "refund",
        "payment",
        "lawsuit",
        "legal",
        "personal data"
    ]

    for word in sensitive_words:
        if word in message_lower:
            return "ESCALATE", f"Sensitive issue detected: {word}"

    # Angry customer
    angry_words = [
        "fuck",
        "fucking",
        "hate",
        "worst",
        "terrible",
        "useless"
    ]

    for word in angry_words:
        if word in message_lower:
            return "ESCALATE", "Customer appears highly frustrated"

    # No historical resolution
    if historical_reply == "No similar historical reply found.":
        return "ESCALATE", "No similar historical resolution found"

    # Unknown intent
    if intent == "general_issue":
        return "ESCALATE", "Intent could not be confidently classified"

    return "AUTO-HANDLE", "Normal issue with known historical resolution"
