import streamlit as st
import pickle
import pandas as pd


with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)


with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)


with open('svc_m.pkl', 'rb') as svc:
    svc_mo = pickle.load(svc)



def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'



def main():
    st.title('Modelado de IRIS dataset')
    st.sidebar.header('Parametros entrada del usuario')

    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
        data = {
            'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width,
            }
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_parameters()

    option = ['Linear Regression','Logistic Regression','SVM']
    model = st.sidebar.selectbox('', option)


    st.subheader('Parametros del usuario')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_mo.predict(df)))

if __name__ == '__main__':
    main()
