from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


# --- Data models -----------------------------------------------------------
@dataclass
class FinancialProfile:
    age: int
    profession: str
    income: float
    address: str
    education: str
    liquid_assets: float
    investment_assets: float
    real_estate_assets: float
    other_assets: float
    mortgage_debt: float
    student_debt: float
    credit_card_debt: float
    other_debt: float
    monthly_essential_expenses: float
    monthly_discretionary_expenses: float
    monthly_insurance_costs: float
    monthly_other_costs: float
    dependents: int
    estate_documents: str
    risk_appetite: str
    llm_notes: str = ""

    @property
    def total_assets(self) -> float:
        return (
            self.liquid_assets
            + self.investment_assets
            + self.real_estate_assets
            + self.other_assets
        )

    @property
    def total_debt(self) -> float:
        return (
            self.mortgage_debt
            + self.student_debt
            + self.credit_card_debt
            + self.other_debt
        )

    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_debt

    @property
    def monthly_income(self) -> float:
        return self.income / 12 if self.income else 0.0

    @property
    def recurring_costs(self) -> float:
        return (
            self.monthly_essential_expenses
            + self.monthly_discretionary_expenses
            + self.monthly_insurance_costs
            + self.monthly_other_costs
        )


@dataclass
class AdvisorBudgetItem:
    category: str
    amount: float
    notes: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AdvisorRetirementPoint:
    age: float
    year: int
    balance: float
    assumptions: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AdvisorCatastrophicPlan:
    scenario: str
    recommended_reserve: float
    guidance: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AdvisorResponse:
    budget: List[AdvisorBudgetItem]
    retirement_projection: List[AdvisorRetirementPoint]
    catastrophic_plan: List[AdvisorCatastrophicPlan]
    narrative: str
    sources: List[str]


@dataclass
class LLMSettings:
    client: Optional[LLMClient]
    model: Optional[str]
    temperature: float
    max_tokens: int


class LLMClient:
    """Minimal client for OpenAI-compatible chat endpoints (vLLM, Ollama, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        endpoint = f"{self.base_url}/v1/chat/completions"
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        return response.json()


def sanitize_number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def extract_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        if text[idx] != "{" and text[idx] != "[":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx += 1
            continue
        return obj
    return None


def parse_advisor_response(raw: Dict[str, Any]) -> AdvisorResponse:
    message = raw.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    parsed = extract_first_json_block(content)
    if not parsed:
        raise ValueError("LLM response did not contain valid JSON.")

    budget_items = [
        AdvisorBudgetItem(
            category=item.get("category", ""),
            amount=sanitize_number(item.get("amount")),
            notes=item.get("notes"),
            source=item.get("source"),
        )
        for item in parsed.get("budget", [])
    ]

    retirement_projection = [
        AdvisorRetirementPoint(
            age=sanitize_number(point.get("age")),
            year=int(sanitize_number(point.get("year"))),
            balance=sanitize_number(point.get("balance")),
            assumptions=point.get("assumptions"),
            source=point.get("source"),
        )
        for point in parsed.get("retirement_projection", [])
    ]

    catastrophic_plan = [
        AdvisorCatastrophicPlan(
            scenario=item.get("scenario", ""),
            recommended_reserve=sanitize_number(item.get("recommended_reserve")),
            guidance=item.get("guidance"),
            source=item.get("source"),
        )
        for item in parsed.get("catastrophic_plan", [])
    ]

    narrative = parsed.get("narrative", "")
    sources = [str(source) for source in parsed.get("sources", [])]

    return AdvisorResponse(
        budget=budget_items,
        retirement_projection=retirement_projection,
        catastrophic_plan=catastrophic_plan,
        narrative=narrative,
        sources=sources,
    )


SYSTEM_PROMPT = """
You are a fiduciary financial analyst tasked with building a personalized savings and
investment plan. Combine the provided household profile with current data from the Bureau
of Labor Statistics (BLS), O*NET, regional cost-of-living indices, and relevant policy or
macroeconomic news. Never fabricate figures; cite every quantitative statement with its
source. Respond in JSON with the structure:
{
  "budget": [
    {"category": str, "amount": number, "notes": str, "source": str}
  ],
  "retirement_projection": [
    {"age": number, "year": int, "balance": number, "assumptions": str, "source": str}
  ],
  "catastrophic_plan": [
    {"scenario": str, "recommended_reserve": number, "guidance": str, "source": str}
  ],
  "narrative": str,
  "sources": [str]
}
Ensure that the calculations reference the cited datasets and that all currency values are
expressed in USD.
"""


def build_user_message(profile: FinancialProfile) -> str:
    profile_payload = {
        "age": profile.age,
        "profession": profile.profession,
        "income": profile.income,
        "address": profile.address,
        "education": profile.education,
        "assets": {
            "liquid": profile.liquid_assets,
            "investment": profile.investment_assets,
            "real_estate": profile.real_estate_assets,
            "other": profile.other_assets,
        },
        "debts": {
            "mortgage": profile.mortgage_debt,
            "student": profile.student_debt,
            "credit_card": profile.credit_card_debt,
            "other": profile.other_debt,
        },
        "monthly_costs": {
            "essential": profile.monthly_essential_expenses,
            "discretionary": profile.monthly_discretionary_expenses,
            "insurance": profile.monthly_insurance_costs,
            "other": profile.monthly_other_costs,
        },
        "dependents": profile.dependents,
        "estate_documents": profile.estate_documents,
        "risk_appetite": profile.risk_appetite,
        "net_worth": profile.net_worth,
        "monthly_income": profile.monthly_income,
        "recurring_costs": profile.recurring_costs,
    }

    request_payload = {
        "profile": profile_payload,
        "analysis_requirements": [
            "Quantify job outlook, wage trends, and employment volatility using BLS and O*NET data.",
            "Adjust cost projections by the household's location using a reputable cost-of-living index.",
            "Account for current economic indicators (inflation, interest rates, policy changes).",
            "Provide recommended monthly budget allocations that match the user's risk appetite.",
            "Project retirement balances with explicit assumptions about contribution amounts and returns.",
            "Size emergency and catastrophic reserves for layoffs, medical events, and dependent support.",
            "Reference every figure with a verifiable source URL or citation identifier.",
        ],
        "additional_context": profile.llm_notes,
        "output_requirements": {
            "currency": "USD",
            "explain_methodology": True,
            "include_sources": True,
        },
    }

    return json.dumps(request_payload, indent=2)


def build_llm_payload(
    profile: FinancialProfile, model: str, temperature: float, max_tokens: int
) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": build_user_message(profile)},
        ],
    }


# --- Streamlit app --------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.header("About this advisor")
    st.sidebar.write(
        """
        Configure a local or hosted large language model endpoint that can retrieve live
        Bureau of Labor Statistics, O*NET, cost-of-living, and policy updates. The model
        should return quantitative recommendations backed by citations.
        """
    )
    st.sidebar.info(
        "All results are educational and should be validated with a licensed professional."
    )


def render_llm_settings() -> LLMSettings:
    st.sidebar.subheader("LLM connection")
    base_url = st.sidebar.text_input(
        "Base URL",
        help="Root URL for an OpenAI-compatible API (for example a vLLM or Ollama server).",
    )
    model = st.sidebar.text_input("Model name", help="Identifier exposed by the selected endpoint.")
    api_key = st.sidebar.text_input(
        "API key",
        type="password",
        help="Optional token if the endpoint requires authentication.",
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values keep outputs closer to sourced data.",
    )
    max_tokens = int(
        st.sidebar.number_input(
            "Max tokens",
            min_value=256,
            max_value=8192,
            value=2048,
            step=256,
            help="Upper bound for response length.",
        )
    )

    client = None
    if base_url and model:
        client = LLMClient(base_url=base_url, model=model, api_key=api_key or None)

    return LLMSettings(client=client, model=model or None, temperature=temperature, max_tokens=max_tokens)


def collect_user_inputs() -> FinancialProfile:
    st.header("Household Profile")
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Age", min_value=18, max_value=100, value=18)
        profession = col2.text_input("Profession")
        income = col1.number_input("Annual Income (USD)", min_value=0, value=0, step=1000)
        address = col2.text_input("City, State (e.g. Austin, TX)")
        education = col1.selectbox(
            "Highest Education Level",
            ["High School", "Associate", "Bachelor's", "Master's", "Doctorate"],
            index=2,
        )

        st.subheader("Assets")
        liquid_assets = st.number_input("Liquid Assets (cash, savings)", min_value=0.0, value=0.0, step=1000.0)
        investment_assets = st.number_input(
            "Investment Assets (retirement, brokerage)", min_value=0.0, value=0.0, step=1000.0
        )
        real_estate_assets = st.number_input(
            "Real Estate Equity", min_value=0.0, value=0.0, step=5000.0
        )
        other_assets = st.number_input("Other Assets", min_value=0.0, value=0.0, step=1000.0)

        st.subheader("Debts")
        mortgage_debt = st.number_input("Mortgage Debt", min_value=0.0, value=0.0, step=5000.0)
        student_debt = st.number_input("Student Loans", min_value=0.0, value=0.0, step=1000.0)
        credit_card_debt = st.number_input("Credit Card Debt", min_value=0.0, value=0.0, step=500.0)
        other_debt = st.number_input("Other Debts", min_value=0.0, value=0.0, step=500.0)

        st.subheader("Recurring Monthly Costs")
        monthly_essential = st.number_input("Essentials (housing, utilities, food)", min_value=0.0, value=0.0, step=100.0)
        monthly_discretionary = st.number_input("Discretionary spending", min_value=0.0, value=0.0, step=100.0)
        monthly_insurance = st.number_input("Insurance premiums", min_value=0.0, value=0.0, step=50.0)
        monthly_other = st.number_input("Other monthly obligations", min_value=0.0, value=0.0, step=50.0)

        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0, step=1)
        estate_documents = st.text_area(
            "Existing wills/trusts (include states where active)",
            value="",
        )
        risk_appetite = st.selectbox("Risk Appetite", ["Careful", "Moderate", "Aggressive"], index=1)

        st.subheader("Additional instructions for the LLM")
        llm_notes = st.text_area(
            "Optional context",
            help=(
                "Share any qualitative information, links to custom research, or prompts the model should"
                " consider when aggregating external datasets."
            ),
        )

        submitted = st.form_submit_button("Generate Plan")

    if not submitted:
        st.stop()

    return FinancialProfile(
        age=age,
        profession=profession,
        income=float(income),
        address=address,
        education=education,
        liquid_assets=float(liquid_assets),
        investment_assets=float(investment_assets),
        real_estate_assets=float(real_estate_assets),
        other_assets=float(other_assets),
        mortgage_debt=float(mortgage_debt),
        student_debt=float(student_debt),
        credit_card_debt=float(credit_card_debt),
        other_debt=float(other_debt),
        monthly_essential_expenses=float(monthly_essential),
        monthly_discretionary_expenses=float(monthly_discretionary),
        monthly_insurance_costs=float(monthly_insurance),
        monthly_other_costs=float(monthly_other),
        dependents=int(dependents),
        estate_documents=estate_documents,
        risk_appetite=risk_appetite,
        llm_notes=llm_notes,
    )


def render_summary(profile: FinancialProfile) -> None:
    st.subheader("Net Worth Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Assets", f"${profile.total_assets:,.0f}")
    col2.metric("Total Debt", f"${profile.total_debt:,.0f}")
    col3.metric("Net Worth", f"${profile.net_worth:,.0f}")

    st.write(
        f"Monthly take-home estimate: **${profile.monthly_income:,.0f}** | Recurring costs: **${profile.recurring_costs:,.0f}**"
    )

    if profile.estate_documents:
        st.caption(f"Estate planning documents noted: {profile.estate_documents}")


def render_budget_section(items: List[AdvisorBudgetItem]) -> None:
    st.header("Dynamic Budget Plan")
    if not items:
        st.warning("No budget recommendations were returned by the model.")
        return

    df = pd.DataFrame(
        {
            "Category": [item.category for item in items],
            "Amount": [item.amount for item in items],
            "Notes": [item.notes or "" for item in items],
            "Source": [item.source or "" for item in items],
        }
    )
    st.dataframe(df.style.format({"Amount": "${:,.2f}"}))
    st.bar_chart(df.set_index("Category")["Amount"])


def render_retirement_section(points: List[AdvisorRetirementPoint]) -> None:
    st.header("Retirement Trajectory")
    if not points:
        st.warning("No retirement projection returned. Confirm the model prompt includes projection requirements.")
        return

    df = pd.DataFrame(
        {
            "Age": [point.age for point in points],
            "Year": [point.year for point in points],
            "Balance": [point.balance for point in points],
            "Assumptions": [point.assumptions or "" for point in points],
            "Source": [point.source or "" for point in points],
        }
    )

    latest = df.iloc[-1]
    st.metric(
        label=f"Projected balance by age {latest['Age']:.0f}",
        value=f"${latest['Balance']:,.0f}",
    )

    st.dataframe(df.style.format({"Balance": "${:,.2f}", "Age": "{:.1f}"}))
    st.line_chart(df.set_index("Age")["Balance"])


def render_catastrophic_section(items: List[AdvisorCatastrophicPlan]) -> None:
    st.header("Catastrophic Event Preparedness")
    if not items:
        st.warning("No catastrophic event planning guidance returned.")
        return

    df = pd.DataFrame(
        {
            "Scenario": [item.scenario for item in items],
            "Recommended Reserve": [item.recommended_reserve for item in items],
            "Guidance": [item.guidance or "" for item in items],
            "Source": [item.source or "" for item in items],
        }
    )
    st.dataframe(df.style.format({"Recommended Reserve": "${:,.2f}"}))


def render_narrative(response: AdvisorResponse) -> None:
    if response.narrative:
        st.header("Narrative Summary")
        st.write(response.narrative)

    if response.sources:
        st.subheader("Citations")
        for source in response.sources:
            st.markdown(f"- {source}")


def main() -> None:
    st.set_page_config(page_title="Personalized Investment Advisor", layout="wide")
    st.title("Personalized Investment & Savings Advisor")
    st.write(
        "Provide your financial profile to generate dynamic budgeting guidance, retirement projections, and contingency planning."
    )

    render_sidebar()
    settings = render_llm_settings()
    profile = collect_user_inputs()
    render_summary(profile)

    if not settings.client or not settings.model:
        st.error("Provide the LLM base URL and model name in the sidebar to generate recommendations.")
        st.stop()

    payload = build_llm_payload(
        profile=profile,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    with st.expander("LLM request payload", expanded=False):
        preview = {"model": payload["model"], "temperature": payload["temperature"], "messages": payload["messages"]}
        st.code(json.dumps(preview, indent=2))

    raw_response: Optional[Dict[str, Any]] = None
    try:
        with st.spinner("Requesting plan from LLM..."):
            raw_response = settings.client.generate_plan(payload)
        advisor_response = parse_advisor_response(raw_response)
    except Exception as exc:  # pragma: no cover - surface error to UI
        st.error(f"Failed to generate plan: {exc}")
        if raw_response is not None:
            with st.expander("Raw LLM response"):
                st.code(json.dumps(raw_response, indent=2))
        st.stop()

    render_budget_section(advisor_response.budget)
    render_retirement_section(advisor_response.retirement_projection)
    render_catastrophic_section(advisor_response.catastrophic_plan)
    render_narrative(advisor_response)


if __name__ == "__main__":
    main()
