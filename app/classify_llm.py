"""
classify_llm.py
---------------
Uses Claude 3.5 Sonnet via OpenRouter to classify transactions intelligently.

It assigns:
- category: [income, housing_rent, tuition, subscription, utilities, food, transport, other]
- is_recurring: true/false
- periodicity: "monthly" or null
- priority: integer 0–5 (lower = more important)

The model learns from your example priorities below and applies them to new transactions.
"""

from __future__ import annotations
import pandas as pd
import json
import os
from openai import OpenAI

# --- Few-shot example based on your CSV ---
EXAMPLE_PRIORITIES = """
Example pattern for priorities:

type,description,amount,date,priority
income,Student Salary,9000,2025-11-01,0
expense,Transport (Car Gas) #1,-1000,2025-11-01,1
expense,Groceries #1,-700,2025-11-03,1
expense,Internet Bill,-389,2025-11-03,2
expense,House System Alarm,-260,2025-11-05,2
expense,Netflix Subscription,-150,2025-11-09,3
expense,Groceries #2,-700,2025-11-10,1
expense,Gym Membership,-600,2025-11-11,2
expense,Water Bill,-120,2025-11-14,1
expense,Transport (Car Gas) #2,-1000,2025-11-15,1
income,Freelance Job,2000,2025-11-16,0
expense,Groceries #3,-700,2025-11-17,1
expense,Electric Bill,-650,2025-11-19,1
expense,Groceries #4,-700,2025-11-24,1
expense,Apple Music,-120,2025-11-27,3
expense,Apple Theft and Loss Coverage,-100,2025-11-29,3
"""

SYSTEM_PROMPT = f"""You are a financial assistant that classifies transactions into logical categories.

For each transaction, assign:
- category: one of [income, housing_rent, tuition, subscription, utilities, food, transport, other]
- is_recurring: true or false
- periodicity: "monthly" if recurring, otherwise null
- priority: integer from 0–5 (lower = more important)

Base your logic on this example priority pattern:
{EXAMPLE_PRIORITIES}

Rules for priority:
- 0 = income (positive amounts)
- 1 = must pay (rent, tuition, essential bills)
- 2 = basic services (internet, utilities)
- 3 = medium importance (subscriptions, insurance)
- 4–5 = optional or luxury

Take as an example this csv:
type,description,amount,date,priority
income,Student Salary,9000,2025-11-01,0
expense,Transport (Car Gas) #1,-1000,2025-11-01,1
expense,Groceries #1,-700,2025-11-03,1
expense,Internet Bill,-389,2025-11-03,2
expense,House System Alarm,-260,2025-11-05,2
expense,Netflix Subscription,-150,2025-11-09,3
expense,Groceries #2,-700,2025-11-10,1
expense,Gym Membership,-600,2025-11-11,2
expense,Water Bill,-120,2025-11-14,1
expense,Transport (Car Gas) #2,-1000,2025-11-15,1
income,Freelance Job,2000,2025-11-16,0
expense,Groceries #3,-700,2025-11-17,1
expense,Electric Bill,-650,2025-11-19,1
expense,Groceries #4,-700,2025-11-24,1
expense,Apple Music,-120,2025-11-27,3
expense,Apple Theft and Loss Coverage,-100,2025-11-29,3


Don't not Output JSON only, with one object per transaction.
Each object must include: description, amount, category, is_recurring, periodicity, priority.
"""

def classify_with_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses Claude 3.5 Sonnet (via OpenRouter) to classify transactions.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Please set it in your environment.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Compact JSON of transactions to classify
    rows = df[["description", "amount"]].to_dict(orient="records")
    prompt = f"Classify the following transactions:\n{json.dumps(rows, indent=2)}"

    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.2  
        )

        raw_text = response.choices[0].message.content.strip()

        # Try parsing the model's JSON output
        data = json.loads(raw_text)
        df_llm = pd.DataFrame(data)

        # Merge back with original df (keeping existing columns)
        merged = pd.concat([df.reset_index(drop=True), df_llm], axis=1)

        return merged

    except Exception as e:
        print(f"⚠️ Error while classifying with LLM: {e}")
        return df
