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

def classify_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the DataFrame from ingest and add:
    - category
    - is_recurring
    - periodicity
    - priority
    - pay_on (if not already present)
    """
    # We make a copy to avoid mutating in place (safer)
    out = df.copy()

    out["category"] = [
        detect_category(desc, amt)
        for desc, amt in zip(out["description"], out["amount"])
    ]

    out["is_recurring"] = [
        infer_recurring(cat, desc)
        for cat, desc in zip(out["category"], out["description"])
    ]

    out["periodicity"] = [infer_periodicity(ir) for ir in out["is_recurring"]]

    out["priority"] = [assign_priority(cat) for cat in out["category"]]

    # Check if 'pay_on' exists. If not, create it from 'date' or use a default date.
    if 'pay_on' not in out.columns:
        if 'date' in out.columns:
            out['pay_on'] = pd.to_datetime(out['date'], errors='coerce')  # Use the 'date' column if available
        else:
            out['pay_on'] = pd.to_datetime('2025-11-04')  # Default date if no date exists

    # Make sure 'pay_on' is in a proper datetime format
    out['pay_on'] = pd.to_datetime(out['pay_on'], errors='coerce')

    return out


if __name__ == "__main__":
    # quick local test:
    from ingest import load_all_transactions
    
    df_raw: pd.DataFrame = load_all_transactions()
    df_classified: pd.DataFrame = classify_transactions(df_raw)
    print(df_classified)
