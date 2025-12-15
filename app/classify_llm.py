from __future__ import annotations
import os
import json
import re
import pandas as pd
from openai import OpenAI

EXAMPLE_PRIORITIES = """
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

SYSTEM_PROMPT = f"""You are a financial assistant that classifies transactions.

For each transaction, you MUST return JSON only, as a list of objects.
Each object must include: description, amount, category, is_recurring, periodicity, priority.

Valid categories: [income, housing_rent, tuition, subscription, utilities, food, transport, other]

Priority rules (MANDATORY):
- 0 = income (positive amounts)
- 1 = must pay (rent, tuition, essential bills like groceries, transport, power, water)
- 2 = basic services (internet, phone, other utilities)
- 3 = medium importance (subscriptions, insurance)
- 4–5 = optional or luxury

Base your behavior on this example:
{EXAMPLE_PRIORITIES}
"""

# --- deterministic local rules to fix priorities ---
def _rule_based_priority(row: pd.Series) -> int:
    desc = str(row.get("description", "")).lower()
    amt = float(row.get("amount", 0) or 0)
    t = str(row.get("type", "")).lower()
    cat = str(row.get("category", "")).lower()

    # income always 0
    if amt > 0 or t == "income" or cat == "income":
        return 0

    # rent / tuition / essentials
    if any(k in desc for k in ["rent", "apartment", "room", "house", "tuition", "colegiatura"]):
        return 1
    if any(k in desc for k in ["groceries", "grocery", "supermarket"]):
        return 1
    if any(k in desc for k in ["transport", "car gas", "gasoline", "uber", "taxi", "bus"]):
        return 1
    if any(k in desc for k in ["electric bill", "electricity", "power", "water bill", "cfe"]):
        return 1

    # basic services (internet / phone)
    if "internet" in desc or "wifi" in desc:
        return 2
    if "phone" in desc or "mobile" in desc or "cell" in desc:
        return 2

    # subscriptions / insurance
    if any(k in desc for k in ["netflix", "spotify", "apple music", "youtube", "membership", "gym"]):
        return 3
    if "insurance" in desc or "coverage" in desc:
        return 3

    # default: optional / luxury
    return 4


def classify_with_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses DeepSeek V3.1 (via OpenRouter) to classify transactions.
    - LLM suggests category / is_recurring / periodicity
    - Final priority is ALWAYS corrected using local rules (_rule_based_priority)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Please set it in your environment.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # only send essential info
    rows = df[["description", "amount"]].to_dict(orient="records")
    prompt = f"Classify the following transactions:\n{json.dumps(rows, indent=2)}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.5,
        )

        raw_text = (response.choices[0].message.content or "").strip()

        # clean code fences like ```json
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(json)?", "", raw_text).strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        # try to parse list JSON
        try:
            data = json.loads(raw_text)
        except Exception:
            m = re.search(r"\[.*\]", raw_text, re.DOTALL)
            if not m:
                print("⚠️ Could not parse LLM JSON, returning original df.")
                return df
            data = json.loads(m.group(0))

        if not isinstance(data, list) or not data:
            print("⚠️ Empty or invalid LLM data, returning original df.")
            return df

        df_llm = pd.DataFrame(data)

        # merge side-by-side by order
        merged = pd.concat([df.reset_index(drop=True), df_llm.reset_index(drop=True)], axis=1)

        # rename duplicate suffixes (_x, _y)
        merged = merged.rename(columns=lambda c: c.replace("_x", "").replace("_y", ""))

        # remove truly duplicated column names to avoid Series ambiguity
        merged = merged.loc[:, ~merged.columns.duplicated()]  

        # always apply local rules for final priority
        merged["priority"] = merged.apply(_rule_based_priority, axis=1).astype(int)

        # ensure final schema
        final_cols = ["id", "type", "description", "amount", "date", "priority", "source", "due_date", "status"]

        # add missing columns as None
        for col in final_cols:
            if col not in merged.columns:
                merged[col] = None

        # keep columns in desired order
        merged = merged[final_cols]

        print("\n🧠 LLM+Rules Classification Preview:")
        print(merged.head(20))

        return merged

    except Exception as e:
        print(f"⚠️ Error while classifying with LLM: {e}")
        return df
