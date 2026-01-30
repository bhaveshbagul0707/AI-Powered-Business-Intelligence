import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Business Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ================= GENAI CLIENT (BACKEND ONLY) =================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def ask_genai(prompt):
    if client is None:
        return "❌ GenAI service not configured. API key missing."

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a senior business data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content

# ================= PDF FUNCTION =================
def generate_pdf(text):
    file_path = "AI_Business_Report.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    y = height - 40
    for line in text.split("\n"):
        c.drawString(40, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 40

    c.save()
    return file_path

# ================= ENHANCED CUSTOM CSS =================
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2530 0%, #252d3d 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(79, 172, 254, 0.2);
    }
    
    .metric-title {
        color: #8b92a7;
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    
    .metric-value {
        color: #4facfe;
        font-size: 36px;
        font-weight: 700;
        margin-top: 5px;
    }
    
    .metric-icon {
        font-size: 40px;
        margin-bottom: 10px;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 40px 0 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 16px;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px 25px;
        border-radius: 12px;
        margin: 30px 0 20px 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .section-header h2 {
        color: white;
        margin: 0;
        font-size: 24px;
    }
    
    /* Chart Containers */
    .chart-container {
        background: #1e2530;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2530 0%, #252d3d 100%);
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* Upload area */
    .uploadedFile {
        border-radius: 10px;
        background: #1e2530;
    }
    
    /* Form styling */
    .stForm {
        background: #1e2530;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Data Table */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 30px;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #8b92a7;
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, #1e2530 0%, #252d3d 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 4px solid #4facfe;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODELS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    reg_model = joblib.load(os.path.join(BASE_DIR, "model", "sales_regression.pkl"))
    clf_model = joblib.load(os.path.join(BASE_DIR, "model", "profit_classifier.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
except:
    st.error("⚠️ Model files not found. Please ensure models are in the correct directory.")

# ================= SESSION STATE =================
if "df" not in st.session_state:
    st.session_state.df = None

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "sales_pred" not in st.session_state:
    st.session_state.sales_pred = None

if "profit_pred" not in st.session_state:
    st.session_state.profit_pred = None

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <h1>📊 AI-Powered Business Intelligence Platform</h1>
    <p>EDA • Machine Learning • Generative AI</p>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR NAV =================
st.sidebar.markdown("### 🧭 Navigation")
section = st.sidebar.radio(
    "",
    ["📤 Upload Data", "📈 EDA Dashboard", "🤖 ML + GenAI"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This platform combines exploratory data analysis, "
    "machine learning predictions, and generative AI insights "
    "to help you make data-driven business decisions."
)

# ================= UPLOAD DATA =================
if section == "📤 Upload Data":
    st.markdown('<div class="section-header"><h2>Upload Your Data</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        file = st.file_uploader(
            "Upload Superstore CSV",
            type=["csv"],
            help="Upload your business data in CSV format"
        )

    if file:
        df = pd.read_csv(file, encoding="ISO-8859-1")
        df.columns = df.columns.str.strip()
        st.session_state.df = df
        st.success("✅ Dataset loaded successfully!")

    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown('<div class="section-header"><h2>Data Overview</h2></div>', unsafe_allow_html=True)
        
        # Key Metrics Row
        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()
        total_orders = len(df)
        avg_discount = df["Discount"].mean() if "Discount" in df.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-icon'>💰</div>
                    <div class='metric-title'>Total Sales</div>
                    <div class='metric-value'>${total_sales:,.0f}</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-icon'>📈</div>
                    <div class='metric-title'>Total Profit</div>
                    <div class='metric-value'>${total_profit:,.0f}</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-icon'>📦</div>
                    <div class='metric-title'>Total Orders</div>
                    <div class='metric-value'>{total_orders:,}</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""<div class='metric-card'>
                    <div class='metric-icon'>🏷️</div>
                    <div class='metric-title'>Avg Discount</div>
                    <div class='metric-value'>{avg_discount:.1%}</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data Preview
        with st.expander("📋 View Data Sample", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Data Info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Dataset Shape:**")
            st.info(f"📊 {df.shape[0]} rows × {df.shape[1]} columns")
        
        with col2:
            st.markdown("**Missing Values:**")
            missing = df.isnull().sum().sum()
            if missing == 0:
                st.success(f"✅ No missing values")
            else:
                st.warning(f"⚠️ {missing} missing values found")

# ================= EDA DASHBOARD =================
if section == "📈 EDA Dashboard":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first in the 'Upload Data' section.")
    else:
        df = st.session_state.df
        
        st.markdown('<div class="section-header"><h2>Exploratory Data Analysis</h2></div>', unsafe_allow_html=True)
        
        # Configure matplotlib style
        plt.style.use('dark_background')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sales_by_cat = df.groupby("Category")["Sales"].sum()
            colors = ['#667eea', '#764ba2', '#f093fb']
            sales_by_cat.plot(kind="bar", ax=ax, color=colors)
            ax.set_title("Sales by Category", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Category", fontsize=12)
            ax.set_ylabel("Sales ($)", fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            profit_by_region = df.groupby("Region")["Profit"].sum()
            colors = ['#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
            profit_by_region.plot(kind="bar", ax=ax, color=colors)
            ax.set_title("Profit by Region", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Region", fontsize=12)
            ax.set_ylabel("Profit ($)", fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional Charts
        st.markdown("<br>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        if "Segment" in df.columns:
            with col3:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                segment_sales = df.groupby("Segment")["Sales"].sum()
                colors = ['#fa709a', '#fee140', '#30cfd0']
                ax.pie(segment_sales, labels=segment_sales.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
                ax.set_title("Sales Distribution by Segment", fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
        
        if "Ship Mode" in df.columns:
            with col4:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                ship_mode = df.groupby("Ship Mode")["Sales"].sum().sort_values()
                colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
                ship_mode.plot(kind="barh", ax=ax, color=colors)
                ax.set_title("Sales by Shipping Mode", fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Sales ($)", fontsize=12)
                ax.set_ylabel("Ship Mode", fontsize=12)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)

# ================= ML + GENAI =================
if section == "🤖 ML + GenAI":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first in the 'Upload Data' section.")
    else:
        df = st.session_state.df
        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()

        st.markdown('<div class="section-header"><h2>Machine Learning Predictions</h2></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("prediction_form"):
                st.markdown("**📊 Enter Prediction Parameters**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    qty = st.number_input("Quantity", min_value=1, max_value=100, value=5, help="Number of items")
                    month = st.slider("Month", min_value=1, max_value=12, value=6, help="Month of the year")
                
                with col_b:
                    disc = st.slider("Discount", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Discount percentage")
                    quarter = st.slider("Quarter", min_value=1, max_value=4, value=2, help="Business quarter")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

        with col2:
            st.markdown("**💡 Tips**")
            st.info(
                "Adjust the parameters to see how different business scenarios "
                "affect your predicted sales and profit margins."
            )

        if submit:
            try:
                X = scaler.transform([[qty, disc, 2024, month, quarter]])
                st.session_state.sales_pred = reg_model.predict(X)[0]
                st.session_state.profit_pred = clf_model.predict(X)[0]
                st.session_state.prediction_done = True
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

        if st.session_state.prediction_done:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><h2>Prediction Results</h2></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    f"""<div class='metric-card'>
                        <div class='metric-icon'>💵</div>
                        <div class='metric-title'>Predicted Sales</div>
                        <div class='metric-value'>${st.session_state.sales_pred:,.2f}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            with col2:
                profit_status = "High Profit" if st.session_state.profit_pred == 1 else "Low Profit"
                profit_icon = "📈" if st.session_state.profit_pred == 1 else "📉"
                profit_color = "#43e97b" if st.session_state.profit_pred == 1 else "#fa709a"
                
                st.markdown(
                    f"""<div class='metric-card'>
                        <div class='metric-icon'>{profit_icon}</div>
                        <div class='metric-title'>Profit Category</div>
                        <div class='metric-value' style='color:{profit_color}'>{profit_status}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><h2>AI-Powered Insights</h2></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔍 Explain Prediction", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        explanation = ask_genai(
                            f"""
                            Explain this prediction in simple business terms.

                            Quantity: {qty}
                            Discount: {disc}
                            Month: {month}
                            Quarter: {quarter}
                            Predicted Sales: {st.session_state.sales_pred}
                            Profit Category: {'High' if st.session_state.profit_pred == 1 else 'Low'}
                            """
                        )
                        st.info(explanation)

            with col2:
                if st.button("💡 Get Recommendations", use_container_width=True):
                    with st.spinner("Generating recommendations..."):
                        recs = ask_genai(
                            f"""
                            Total Sales: {total_sales}
                            Total Profit: {total_profit}
                            Predicted Sales: {st.session_state.sales_pred}

                            Provide 3 actionable business recommendations.
                            """
                        )
                        st.success(recs)

            with col3:
                if st.button("📄 Generate Report", use_container_width=True):
                    with st.spinner("Creating comprehensive report..."):
                        report = ask_genai(
                            f"""
                            Generate a business report with:
                            - Executive Summary
                            - Key Insights
                            - Risks
                            - Recommendations

                            Total Sales: {total_sales}
                            Total Profit: {total_profit}
                            Predicted Sales: {st.session_state.sales_pred}
                            """
                        )

                        st.text_area("📋 AI Generated Report", report, height=300)

                        pdf = generate_pdf(report)
                        with open(pdf, "rb") as f:
                            st.download_button(
                                "⬇️ Download PDF Report",
                                f,
                                file_name="Business_Report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

# ================= FOOTER =================
st.markdown("""
<div class="custom-footer">
    <p>Built with 💙 using Python • Machine Learning • GenAI • Streamlit</p>
    <p style="font-size: 12px; margin-top: 10px;">© 2025 AI Business Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)