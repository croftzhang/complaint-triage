"""Product taxonomy: map CFPB raw product names to 9 stable groups.

CFPB renamed several products over the years; both old and new names map to
one group so that the classifier label set stays fixed.
"""
from __future__ import annotations

PRODUCT_GROUPS: dict[str, list[str]] = {
    "Credit reporting": [
        "Credit reporting or other personal consumer reports",
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Credit reporting",
    ],
    "Debt collection": ["Debt collection"],
    "Mortgage": ["Mortgage"],
    "Credit card": ["Credit card", "Credit card or prepaid card", "Prepaid card"],
    "Checking or savings account": ["Checking or savings account", "Bank account or service"],
    "Student loan": ["Student loan"],
    "Vehicle loan or lease": ["Vehicle loan or lease", "Consumer Loan"],
    "Money transfer or service": [
        "Money transfer, virtual currency, or money service",
        "Money transfers",
        "Virtual currency",
    ],
    "Payday or personal loan": [
        "Payday loan, title loan, personal loan, or advance loan",
        "Payday loan, title loan, or personal loan",
        "Payday loan",
    ],
}

LABELS: list[str] = list(PRODUCT_GROUPS.keys())
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

RAW_TO_GROUP: dict[str, str] = {
    raw: group for group, raws in PRODUCT_GROUPS.items() for raw in raws
}

# Product groups where Regulation E (electronic fund transfers) is the primary
# regulation. Other groups are routed to a human without the Reg E RAG step.
REG_E_SCOPE: set[str] = {"Checking or savings account", "Money transfer or service"}

MONETARY_RELIEF = "Closed with monetary relief"


def to_group(raw_product: str | None) -> str | None:
    if raw_product is None:
        return None
    return RAW_TO_GROUP.get(raw_product.strip())
