import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Visualizing the data")

df=pd.read_csv("data/Bank_Personal_Loan_Modelling.csv")
st.write("# First 5 rows of the dataset are:")
st.dataframe(df.head(5))

st.write("# Let's get a brief info about the dataset:")
import io
import sys
buffer=io.StringIO()
df.info(buf=buffer)
s=buffer.getvalue()
st.text(s)

st.write("# And, here is Description of the dataset:")
st.dataframe(df.describe())

st.write("# Education of people as given:")
st.dataframe(df['Education'].value_counts())

st.subheader("# Labels of Education are:")
st.markdown("""
            -**1**:Undergraduate\n
            -**2**:Graduate\n
            -**3**:Advanced/Higher\n
            """)

# Replace education codes with labels temporarily for better plots
edu_map = {1: 'Undergraduate', 2: 'Graduate', 3: 'Advanced/Higher'}
df['Education Label'] = df['Education'].map(edu_map)

st.subheader("🎓 Education Level Distribution")
sns.countplot(x='Education Label', data=df)
plt.title("Distribution of Education Levels")
st.pyplot(plt)
plt.clf()

st.subheader("# Target Variable (Personal Loan) Distribution")
sns.countplot(x='Personal Loan', data=df)
st.pyplot(plt)
plt.clf()

