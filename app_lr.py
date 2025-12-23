import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error,confusion_matrix, roc_curve, auc, accuracy_score,mean_squared_error,r2_score

st.set_page_config("Linear Regression", layout="centered")

#load css
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
#title
st.markdown("""
<div class="card">
<h1>Linear Regression</h1>
<p> <center>predict outcomes using linear regression </center></p>
</div>
""", unsafe_allow_html=True)

#Load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

#dataset preview
st.markdown('<div class="card"', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

#prepare data
X=df[["total_bill"]]
y=df["tip"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


#metrics
mae=mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2= 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)






#visualization

st.markdown('<div class= "card">', unsafe_allow_html= True)
st.subheader("Toatl bill vs Tip")
fig,ax =plt.subplots()
ax.scatter(df["total_bill"], df ["tip"], alpha= 0.6)
# create a regression line: sort X so x and y have matching dimensions
sorted_idx = X["total_bill"].argsort()
X_sorted = X.iloc[sorted_idx]
y_pred_all = model.predict(scaler.transform(X_sorted))
ax.plot(X_sorted["total_bill"].values, y_pred_all, color="red", label="Regression line")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html =True)

# #display metrics
st.markdown('<div class= "card">"', unsafe_allow_html =True)
st.subheader("Model Performance")
c1,c2 = st.columns(2)
c1.metric("MAE", f"{mae: .2f}") 
c2.metric("RMSE", f"{mse: .2f}") 
c3,c4 =st.columns(2) 
c3.metric("R2",f"{r2: .3f}") 
c4.metric("adj R2",f"{adj_r2: .3f}") 
st.markdown('</div>', unsafe_allow_html =True)


# model coefficients
st.markdown(f"""
            <div class= "card">
            
            <h3>model intercept and coefficient</h3>
            <p>intercept: {model.intercept_: .3f}</p>
            <p>coefficient for total_bill: {model.coef_[0]: .3f}</p>   
            </div>
            """, unsafe_allow_html =True)
#predict new data
st.markdown('<div class= "card">"', unsafe_allow_html =True)
st.subheader("Predict Tip for New  Bill")
new_total_bill = st.slider("Total Bill Amount", float(df["total_bill"].min()), float(df["total_bill"].max()),30.0)
tip=model.predict(scaler.transform([[new_total_bill]]))
st.markdown(f"<div class='prediction-box'> Predicted Tip: {tip[0]: .2f}</div>", unsafe_allow_html=True)































