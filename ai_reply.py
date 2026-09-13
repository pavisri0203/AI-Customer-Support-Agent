from openai import OpenAI
from classifier import classify_intent
from rag import get_historical_reply
from escalation import check_escalation

client = OpenAI()

message = input("Customer message: ")

# 1. Classify intent
intent = classify_intent(message)

# 2. Get historical resolution
historical_reply = get_historical_reply(message)

# 3. Check escalation
decision, reason = check_escalation(
    message,
    intent,
    historical_reply
)

# 4. Generate AI reply
prompt = f"""
You are an Apple Support customer service assistant.

Write a short, polite and helpful reply.

Ground your response in the historical Apple Support response.
Do not invent policies, refunds, guarantees, or unsupported technical steps.

Customer message:
{message}

Intent:
{intent}

Historical Apple Support response:
{historical_reply}

Generate only the final customer reply.
"""

response = client.responses.create(
    model="gpt-5.6-luna",
    input=prompt
)

print("\nDetected Intent:", intent)

print("\nHistorical AppleSupport Reply:")
print(historical_reply)

print("\nAI Generated Reply:")
print(response.output_text)

print("\nDecision:", decision)
print("Reason:", reason)
