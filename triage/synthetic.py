"""Synthetic complaint generator for the smoke test (no network needed).

The narratives are templated and NOT real CFPB data; they only exercise the
code path. Use scripts/01_download_cfpb.py for real data.
"""
from __future__ import annotations

import random

import pandas as pd

from .taxonomy import LABELS, MONETARY_RELIEF, PRODUCT_GROUPS

_TEMPLATES: dict[str, list[str]] = {
    "Credit reporting": [
        "There is an account on my credit report that does not belong to me. I disputed it with the bureau but it is still there.",
        "My credit report shows a late payment from {month} that I never missed. The dispute was closed without investigation.",
        "An inquiry appeared on my report from a lender I never applied to. I want it removed and my score corrected.",
    ],
    "Debt collection": [
        "A collection agency keeps calling about a debt of ${amount} that I do not owe. They refuse to send validation.",
        "I asked the collector in writing to stop contacting me but they called my workplace again in {month}.",
        "The debt collector reported a balance of ${amount} that was already paid in full years ago.",
    ],
    "Mortgage": [
        "My mortgage servicer misapplied my {month} payment of ${amount} and charged a late fee.",
        "The escrow analysis increased my payment by ${amount} with no explanation and the servicer will not respond.",
        "I applied for a loan modification months ago and keep getting asked for the same documents.",
    ],
    "Credit card": [
        "My credit card was charged ${amount} for a purchase I did not make. The issuer denied my dispute without explanation.",
        "The card company raised my interest rate without notice and refuses to reverse ${amount} in interest.",
        "I closed the card in {month} but the issuer keeps billing an annual fee of ${amount}.",
    ],
    "Checking or savings account": [
        "Someone used my debit card for ${amount} in ATM withdrawals I did not authorize. I reported it in {month} and the bank refused a refund saying I reported too late.",
        "An unauthorized electronic transfer of ${amount} left my checking account. The bank has not completed its investigation after many weeks and gave no provisional credit.",
        "The bank charged overdraft fees of ${amount} on debit card transactions even though I never opted in to overdraft coverage.",
        "A preauthorized payment of ${amount} was taken from my savings after I asked the bank to stop it three days before the date.",
    ],
    "Student loan": [
        "My student loan servicer lost my income-driven repayment application and put me in forbearance without consent.",
        "I made extra payments of ${amount} toward principal but the servicer applied them to future interest.",
        "The servicer reported my loan as delinquent in {month} even though it was in deferment.",
    ],
    "Vehicle loan or lease": [
        "The auto lender repossessed my car in {month} even though my payment of ${amount} had cleared.",
        "My vehicle loan payoff amount changed by ${amount} between two statements and nobody can explain why.",
        "The dealer added a warranty of ${amount} to my auto loan that I declined in writing.",
    ],
    "Money transfer or service": [
        "I sent ${amount} through a remittance app in {month} and the money never arrived. The company says the transfer completed but the recipient got nothing.",
        "A peer to peer transfer of ${amount} was sent to the wrong person because of an app error and the company will not investigate.",
        "I cancelled an international money transfer within minutes but the company kept a fee of ${amount}.",
    ],
    "Payday or personal loan": [
        "The payday lender debited my account for ${amount} twice in {month} and refuses to refund the duplicate.",
        "My personal loan shows a balance of ${amount} higher than the contract because of fees I was never told about.",
        "The lender keeps renewing my title loan with new fees of ${amount} without my permission.",
    ],
}

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
_VIA = ["Web", "Web", "Web", "Referral", "Phone", "Postal mail"]
_TAGS = [None, None, None, None, "Older American", "Servicemember", "Older American, Servicemember"]
_RESPONSES = [
    "Closed with explanation",
    "Closed with explanation",
    "Closed with explanation",
    "Closed with non-monetary relief",
    MONETARY_RELIEF,
]


def make_synthetic_complaints(n_per_class: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    cid = 100000
    for group in LABELS:
        raw_names = PRODUCT_GROUPS[group]
        for _ in range(n_per_class):
            template = rng.choice(_TEMPLATES[group])
            amount = rng.choice([45, 120, 250, 480, 800, 1500, 2600, 5000])
            narrative = template.format(amount=amount, month=rng.choice(_MONTHS))
            narrative = narrative + " " + rng.choice([
                "I want this resolved.",
                "Please help me get my money back.",
                "This has caused me a lot of stress.",
                "I have attached all the documents.",
            ]) + f" Reference {cid}."
            # Make relief slightly more likely for account/transfer complaints with larger amounts,
            # so the TF model has a (weak) signal to learn.
            relief_bias = 0.15 if group in ("Checking or savings account", "Money transfer or service", "Credit card") else 0.0
            response = MONETARY_RELIEF if rng.random() < relief_bias + amount / 20000 else rng.choice(_RESPONSES)
            rows.append(
                dict(
                    complaint_id=str(cid),
                    date_received=f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                    product=rng.choice(raw_names),
                    sub_product=None,
                    issue=None,
                    sub_issue=None,
                    narrative=narrative,
                    company="Synthetic Bank",
                    state=rng.choice(["NC", "NY", "TX", "CA"]),
                    tags=rng.choice(_TAGS),
                    submitted_via=rng.choice(_VIA),
                    company_response=response,
                    timely="Yes",
                    consumer_disputed=None,
                )
            )
            cid += 1
    return pd.DataFrame(rows)
