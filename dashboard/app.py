import streamlit as st
import pandas as pd

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS (LIGHT PURPLE UI)
# -----------------------------------
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #f5f3ff;
}

/* Headers */
h1, h2, h3 {
    color: #5b21b6;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: #ede9fe;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #c4b5fd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ede9fe;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: #4c1d95;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Buttons */
.stButton>button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
}

.stButton>button:hover {
    background-color: #5b21b6;
    color: white;
}

/* Input Boxes */
.stNumberInput input {
    border-radius: 10px;
}

/* Alert Boxes */
.stAlert {
    border-radius: 12px;
}
            
/* Sidebar dropdown cursor fix */
div[data-baseweb="select"] > div {
    cursor: pointer !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("Customer Analytics Dashboard")

st.write("AI-Powered Customer Intelligence System")

# -----------------------------------
# LOAD DATA
# -----------------------------------
rfm = pd.read_csv(
    "data/online_retail.csv/rfm_segmented.csv"
)

clv = pd.read_csv(
    "data/online_retail.csv/customer_clv.csv"
)

predictions = pd.read_csv(
    "data/purchase_predictions.csv"
)

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Choose Section",
    [
        "Overview",
        "Segments",
        "CLV",
        "Predictions",
        "Customer Lookup",
        "Explainability"
    ]
)

# -----------------------------------
# OVERVIEW PAGE
# -----------------------------------
if page == "Overview":

    st.header("Business Overview")

    total_customers = rfm.shape[0]

    avg_monetary = rfm['Monetary'].mean()

    avg_recency = rfm['Recency'].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Average Spend",
        f"${avg_monetary:.2f}"
    )

    col3.metric(
        "Average Recency",
        f"{avg_recency:.1f} days"
    )

# -----------------------------------
# SEGMENTS PAGE
# -----------------------------------
elif page == "Segments":

    st.header("Customer Segments")

    segment_counts = rfm['Segment'].value_counts()

    st.bar_chart(segment_counts)

    st.subheader("Sample Customers")

    st.dataframe(
        rfm[['Customer ID', 'Segment']].head(20)
    )

# -----------------------------------
# CLV PAGE
# -----------------------------------
elif page == "CLV":

    st.header("Customer Lifetime Value")

    top_clv = clv.sort_values(
        'predicted_clv',
        ascending=False
    ).head(10)

    st.subheader("Top Future Value Customers")

    st.dataframe(
        top_clv[['predicted_clv']]
    )

# -----------------------------------
# PREDICTIONS PAGE
# -----------------------------------
elif page == "Predictions":

    st.header("Purchase Propensity")

    st.bar_chart(
        predictions['Purchase_Probability']
    )

    st.subheader("Prediction Samples")

    st.dataframe(
        predictions.head(20)
    )

# -----------------------------------
# CUSTOMER LOOKUP PAGE
# -----------------------------------
elif page == "Customer Lookup":

    st.header("Customer Lookup")

    customer_id = st.number_input(
        "Enter Customer ID",
        step=1
    )

    customer_data = rfm[
        rfm['Customer ID'] == customer_id
    ]

    if not customer_data.empty:

        st.success("Customer Found")

        st.dataframe(customer_data)

    else:

        st.warning("Customer not found.")

# -----------------------------------
# EXPLAINABILITY PAGE
# -----------------------------------
elif page == "Explainability":

    st.header("Model Explainability")

    st.info(
        "Recency was strongest predictor of future purchase."
    )

    st.write(
        "Customers with lower recency and higher frequency had highest purchase probability."
    )