import streamlit as st
st.set_page_config(page_title="Loan Prediction Web-App")

st.title("🏦 Loan Outcome Prediction/Classification")

st.markdown("""
            What have I got here for y'all?\n
            Use the sidebar to navigate:\n
            -🧠 Explore/Peek into the dataset\n
            -📊 View Logistic Regression model performance\n
            -🌲 Checkout Random Forest performance (with a tree)\n
            -🖋️ Make a loan prediction yourself
            """ )

st.markdown("---")

st.markdown(
    """<div style='text-align: center; color: teal; font_size:16px;'><strong>Made with ❤️ by Anuditya Gupttaa</strong><br>  First deployed project | OL Cohort 1.0</div>""",
    unsafe_allow_html=True
)