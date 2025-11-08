"""
classify.py
-----------
This file takes the raw/normalized transactions DataFrame (the one that
comes from `ingest.py`) and adds extra columns so the planner can make
better decisions.

We do *very simple* rule-based classification:
- look at the description (lowercased)
- check for some keywords (rent, tuition, netflix, spotify, phone, etc.)
- assign:
    category
    is_recurring
    periodicity
    priority

Later we could replace this with an LLM or smarter matcher,
but for the MVP this is enough.
"""

from __future__ import annotations
from typing import Optional

import pandas as pd

# small keyword dictionaries
RENT_KEYWORDS = ["rent", "apartment", "room", "house"]
TUITION_KEYWORDS = ["tuition", "university", "college", "colegiatura"]
SUBSCRIPTION_KEYWORDS = ["spotify", "netflix", "youtube", "membership", "gym"]
UTILITIES_KEYWORDS = ["electricity", "water", "internet", "phone", "cfe"]
FOOD_KEYWORDS = ["groceries", "supermarket", "cafeteria", "food", "restaurant"]
TRANSPORT_KEYWORDS = ["uber", "taxi", "bus"]

def detect_category(description: str, amount: float) -> str:
    """
    Try to map a human description to a high-level spending/income category.
    """
    
    desc = description.lower()

    if amount > 0:
        return "income"
    
    if any(word in desc for word in RENT_KEYWORDS):
        return "housing_rent"
    
    if any(word in desc for word in TUITION_KEYWORDS):
        return "tuition"
    
    if any(word in desc for word in SUBSCRIPTION_KEYWORDS):
        return "subscription"
    
    if any(word in desc for word in UTILITIES_KEYWORDS):
        return "utilities"
    
    if any(word in desc for word in FOOD_KEYWORDS):
        return "food"
    
    if any(word in desc for word in TRANSPORT_KEYWORDS):
        return "transport"

    return "other"


def infer_recurring(category: str, description: str) -> bool:
    """
    Very naive recurring detector.
    Monthly-like stuff = True.
    """
    
    desc = description.lower()

    if category in ("housing_rent", "tuition", "subscription", "utilities"):
        return True

    if "monthly" in desc or "mensual" in desc:
        return True
    
    return False

def infer_periodicity(is_recurring: bool) -> Optional[str]:
    """
    For now we only support 'monthly' as periodicity.
    """

    if is_recurring:
        return "monthly"

    return None


def assign_priority(category: str) -> int:
    """
    Lower number = more important.
    0 = income (not a payment)
    1 = must pay or you get in trouble (rent, tuition)
    2 = basic services
    3 = variable but necessary
    4 = can be postponed
    5 = everything else    
    """

    if category == "income":
        return 0
    
    if category in ("housing_rent", "tuition"):
        return 1
    
    if category == "utilities":
        return 2
    
    if category in ("food", "transport"):
        return 3
    
    if category == "subscription":
        return 4
    
    return 5


ALLOWED_COLS = [
    "id",
    "type",
    "description",
    "amount",
    "date",
    "priority",
    "source",
    "due_date",
    "status",
]


def classify_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and assign priority based on category rules.

    - If 'type' is missing, infer it from amount (>0 income, else expense).
    - For each row, compute category (using detect_category) and then
      priority (using assign_priority).
    - Keep ONLY the allowed columns in the final DataFrame.
    """
    out = df.copy()

    if "type" not in out.columns and "amount" in out.columns:
        out["type"] = out["amount"].apply(
            lambda x: "income" if x > 0 else "expense"
        )

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    priorities: list[int] = []
    for desc, amt, typ in zip(
        out.get("description", []),
        out.get("amount", []),
        out.get("type", []),
    ):
        category = detect_category(str(desc), float(amt))
        pr = assign_priority(category)
        priorities.append(pr)

    out["priority"] = priorities

    cols = [c for c in ALLOWED_COLS if c in out.columns]
    return out[cols]


if __name__ == "__main__":
    # quick local test:
    from ingest import load_all_transactions
    
    df_raw: pd.DataFrame = load_all_transactions()
    df_classified: pd.DataFrame = classify_transactions(df_raw)
    print(df_classified)
