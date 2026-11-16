
import streamlit as st
import joblib
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #f7f4ff;
}

.main-title {
    text-align: center;
    color: #6d28d9;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 30px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(109,40,217,0.10);
    border: 1px solid #e9d5ff;
}

.result-spam {
    color: #dc2626;
    font-size: 30px;
    font-weight: bold;
}

.result-safe {
    color: #16a34a;
    font-size: 30px;
    font-weight: bold;
}

.section-title {
    color: #6d28d9;
    font-size: 22px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background-color: #5b21b6;
    color: white;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "email_input" not in st.session_state:
    st.session_state.email_input = ""


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">📧 Email Spam Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Email Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with ((st.sidebar)):
    st.title("📩 About the System")

    st.write("""
    This system uses Machine Learning to
    identify whether an email is:

    - 🟢 Not Spam (Legitimate Email)
    - 🔴 Spam (Unwanted Email)

    The system uses TF-IDF for text
    feature extraction and Multinomial
    Naive Bayes for classification.
    """)

    st.divider()

    st.subheader("📊 Model Information")

    st.write("**Algorithm:** Naive Bayes")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Training Dataset:** 53,668 original records")
    st.write("**Additional Training:** 200 synthetic legitimate examples")

    st.divider()

    st.caption("Artificial Intelligence 6.0")


# ==========================================
# EMAIL CLASSIFICATION SECTION
# ==========================================

st.markdown(
    '<div class="section-title">🔍 Check Your Email</div>',
    unsafe_allow_html=True
)

st.write("Enter an email message below to determine whether it is spam or legitimate.")

# Sample emails

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Load Legitimate Email", use_container_width=True):
        st.session_state.email_input = (
            "Hello John, good morning. "
            "I would like to schedule a meeting tomorrow "
            "to discuss our group project. "
            "Thank you and have a nice day."
        )

with col2:
    if st.button("🔴 Load Spam Email", use_container_width=True):
        st.session_state.email_input = (
            "Congratulations! You have won a free cash prize. "
            "Click here now to claim your reward!"
        )

email_text = st.text_area(
    "Email Message",
    key="email_input",
    height=180,
    placeholder="Paste or type your email message here..."
)

st.write("")

if st.button("🔍 CHECK EMAIL", use_container_width=True):

    if not email_text.strip():
        st.warning("Please enter an email message first.")

    else:
        try:
            prediction = model.predict([email_text])[0]
            probabilities = model.predict_proba([email_text])[0]

            confidence = max(probabilities) * 100

            # Assuming 0 = Not Spam and 1 = Spam
            if prediction == 1:
                result = "SPAM"
                result_class = "result-spam"
                result_icon = "🚨"
            else:
                result = "NOT SPAM"
                result_class = "result-safe"
                result_icon = "✅"

            # Display result
            st.divider()
            st.subheader("📋 Classification Result")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="{result_class}">
                        {result_icon} {result}
                    </div>
                    <p style="color:#6b7280;">
                        Prediction Confidence
                    </p>
                    <h1 style="color:#6d28d9;">
                        {confidence:.2f}%
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence / 100)

            if prediction == 1:
                st.error("Warning: This email has been classified as spam.")
            else:
                st.success("This email has been classified as legitimate.")

            # Save prediction to history
            st.session_state.history.insert(0, {
                "Email": email_text,
                "Prediction": result,
                "Confidence": f"{confidence:.2f}%"
            })

        except Exception as error:
            st.error(f"Prediction error: {error}")


# ==========================================
# PREDICTION HISTORY
# ==========================================

st.divider()

st.markdown(
    '<div class="section-title">📜 Prediction History</div>',
    unsafe_allow_html=True
)

st.write("Previously checked email messages appear here.")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No prediction history yet. Check an email to get started.")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; color:#777; padding:15px;">
    <b>Email Spam Detection System</b><br>
    Artificial Intelligence 6.0<br>
    Powered by Machine Learning and Streamlit
</div>
""", unsafe_allow_html=True)
