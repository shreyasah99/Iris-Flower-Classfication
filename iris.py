# importing libraries
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier


# html ,css
st.markdown('<style>body{background-color: #9171eb;}</style>',unsafe_allow_html=True)
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#d0c4f2,#d0c4f2);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

#title
html_temp = """
<div style="background-color:#17034f;padding:10px">
<h2 style="color:white;text-align:center;font-size:45px">Iris flowers Classification </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.write("   ")
st.write("  ")





# data-preprocessing
df = pd.read_csv('Iris.csv')
df.drop(['Id'] , axis=1 , inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species']=le.fit_transform(df['Species'])

if st.checkbox("Show dataframe"):
    df


# taking inputs
st.sidebar.header('Features')

a = st.sidebar.slider(" Enter Sepal Length", float(df['SepalLengthCm'].min()) , float(df['SepalLengthCm'].max()) )
b = st.sidebar.slider(" Enter Sepal Width ", float(df['SepalWidthCm'].min()) , float(df['SepalWidthCm'].max()) )
c = st.sidebar.slider(" Enter Petal Length", float(df['PetalLengthCm'].min()) , float(df['PetalLengthCm'].max()) )
d = st.sidebar.slider(" Enter Petal Width", float(df['PetalWidthCm'].min()) , float(df['PetalWidthCm'].max()) )


# building model
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3)
model = DecisionTreeClassifier(criterion = 'gini')
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_train, y_train)

img1 = Image.open("C:\\Users\\shrey\\Downloads\\sentosa.jpg")
img2 = Image.open("C:\\Users\\shrey\\Downloads\\versicolor.jpg")
img3 = Image.open("C:\\Users\\shrey\\Downloads\\virginia.jpg")


# prediction for single input
prediction = model.predict([[a,b,c,d]])
if st.sidebar.button('RUN'):
   if prediction == 0:
       st.image(img1, width=450)
       st.write("It is Sentosa")
   elif prediction == 1:
       st.image(img2, width=450)
       st.write("It is Versicolor")
   elif prediction == 2:
       st.image(img3, width=450)
       st.write("It is virginia")