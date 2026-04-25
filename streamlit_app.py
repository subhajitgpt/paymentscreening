import streamlit as st
import json
from agentic_payment_agent import screen_payment_agentic

st.set_page_config(page_title="Payment Screening (Agentic)", layout="wide")
st.title("Payment Screening For Compliance (Agentic Flow)")
st.markdown("""
Sample data → Algorithms → Scoring → Automated decision → Explanation  
Batch jobs + audit trail
""")

# --- Sidebar: Load test scenarios ---
import os

def load_test_scenarios():
    base_dir = os.path.dirname(__file__)
    payload_dir = os.path.join(base_dir, "test_payloads_flask_api")
    out = {}
    if os.path.isdir(payload_dir):
        for fname in os.listdir(payload_dir):
            if fname.startswith("payload_") and fname.endswith(".json"):
                scenario_name = fname[len("payload_") : -len(".json")]
                fpath = os.path.join(payload_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, dict):
                        out[scenario_name] = payload
                except Exception:
                    continue
    if out:
        return out
    # Fallback to legacy single file.
    path = os.path.join(base_dir, "test_payloads.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scenarios = data.get("test_scenarios") or {}
        for name, entry in scenarios.items():
            if isinstance(entry, dict) and isinstance(entry.get("payload"), dict):
                out[name] = entry["payload"]
        return out
    except Exception:
        return {}

scenarios = load_test_scenarios()
scenario_names = list(scenarios.keys())

with st.sidebar:
    st.header("Sample Scenarios")
    selected_scenario = st.selectbox("Choose a scenario", ["(manual entry)"] + scenario_names)
    if selected_scenario != "(manual entry)":
        payload = scenarios[selected_scenario].copy()
    else:
        payload = {}

# --- Main form ---
def get_form_defaults(payload):
    return {
        "payer_name": payload.get("payer_name", "Global Trade LLC"),
        "payer_address": payload.get("payer_address", "PO Box 12345, Dubai, UAE"),
        "payer_country": payload.get("payer_country", "AE"),
        "payer_dob": payload.get("payer_dob", ""),
        "benef_name": payload.get("benef_name", "Olena Petrenko-Kovalenko"),
        "benef_address": payload.get("benef_address", "Kreschatyk 22, Kyiv, Ukraine"),
        "benef_country": payload.get("benef_country", "UA"),
        "benef_dob": payload.get("benef_dob", "1990-02-01"),
        "amount": payload.get("amount", 12500.0),
        "currency": payload.get("currency", "USD"),
        "reference": payload.get("reference", "Invoice 2025-10-ACME"),
    }

with st.form("screening_form"):
    st.subheader("Single Transaction Screening")
    form_data = get_form_defaults(payload)
    cols = st.columns(2)
    with cols[0]:
        payer_name = st.text_input("Payer Name", form_data["payer_name"])
        payer_address = st.text_input("Payer Address", form_data["payer_address"])
        payer_country = st.text_input("Payer Country", form_data["payer_country"])
        payer_dob = st.text_input("Payer DOB (YYYY-MM-DD)", form_data["payer_dob"])
    with cols[1]:
        benef_name = st.text_input("Beneficiary Name", form_data["benef_name"])
        benef_address = st.text_input("Beneficiary Address", form_data["benef_address"])
        benef_country = st.text_input("Beneficiary Country", form_data["benef_country"])
        benef_dob = st.text_input("Beneficiary DOB (YYYY-MM-DD)", form_data["benef_dob"])
    cols2 = st.columns(3)
    with cols2[0]:
        amount = st.text_input("Amount", form_data["amount"])
    with cols2[1]:
        currency = st.text_input("Currency", form_data["currency"])
    with cols2[2]:
        reference = st.text_input("Reference", form_data["reference"])
    submitted = st.form_submit_button("Run Agentic Screening")

if submitted:
    input_payload = {
        "payer_name": payer_name,
        "payer_address": payer_address,
        "payer_country": payer_country,
        "payer_dob": payer_dob,
        "benef_name": benef_name,
        "benef_address": benef_address,
        "benef_country": benef_country,
        "benef_dob": benef_dob,
        "amount": amount,
        "currency": currency,
        "reference": reference,
    }
    with st.spinner("Screening in progress..."):
        result = screen_payment_agentic(input_payload)
    st.subheader("Result")
    dec = result.get("decision", "")
    if dec == "ESCALATE":
        st.error(f"Decision: {dec}")
    elif dec == "RELEASE":
        st.success(f"Decision: {dec}")
    else:
        st.warning(f"Decision: {dec}")
    st.write(f"**Reason:** {result.get('reason','')}")
    st.write(f"**Best Role:** {result.get('best_role','—')}")
    st.write(f"**Best Score:** {result.get('best_score', 0.0):.3f}")
    st.write(f"**Sanctions Flag:** {'Yes' if result.get('sanction_flag') else 'No'}")
    st.write("**Explanation:**")
    st.code(result.get("explanation", ""), language="text")
    with st.expander("Show full result JSON"):
        st.json(result)

# --- Batch Screening ---
st.subheader("Batch Screening (Upload JSON)")
batch_file = st.file_uploader("Upload batch JSON file", type=["json"])
if batch_file is not None:
    try:
        items = json.load(batch_file)
        if isinstance(items, dict) and "items" in items:
            items = items["items"]
        elif isinstance(items, dict) and "test_scenarios" in items:
            items = [v["payload"] for v in items["test_scenarios"].values() if isinstance(v, dict) and "payload" in v]
        assert isinstance(items, list)
        st.info(f"Loaded {len(items)} items. Running batch screening...")
        results = []
        for idx, item in enumerate(items):
            with st.spinner(f"Screening item {idx+1}/{len(items)}..."):
                res = screen_payment_agentic(item)
                results.append(res)
        st.success(f"Batch screening complete. {len(results)} items processed.")
        for idx, res in enumerate(results):
            with st.expander(f"Result {idx+1}"):
                st.write(f"**Decision:** {res.get('decision','')}")
                st.write(f"**Reason:** {res.get('reason','')}")
                st.write(f"**Best Role:** {res.get('best_role','—')}")
                st.write(f"**Best Score:** {res.get('best_score', 0.0):.3f}")
                st.write(f"**Sanctions Flag:** {'Yes' if res.get('sanction_flag') else 'No'}")
                st.code(res.get("explanation", ""), language="text")
                # Show JSON directly, not in a nested expander
                st.write("**Full result JSON:**")
                st.json(res)
    except Exception as e:
        st.error(f"Failed to process batch: {e}")
