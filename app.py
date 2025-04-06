import streamlit as st
import pickle as pk
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 
import seaborn as sns
import matplotlib.pyplot as plt


# App configuration
st.set_page_config(page_title="Mushromms Classification", layout="wide")

@st.cache_data
def load_data():
  with open("split_data.pkl", "rb") as f:
     X_train, X_test, y_train, y_test = pk.load(f)
  return X_train, X_test, y_train, y_test

class_names = ['edible','poisonous']

X_train, X_test, y_train, y_test = load_data()

def main():

    st.title('Mushromms Classification')
    st.markdown('Are your mushromms edible or poisonous?🍄')
    st.sidebar.title('Sitting 💡')

    def plot_metrics(metrics_list):

        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            # Create Seaborn heatmap
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names,
                          yticklabels=class_names, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            # Display in Streamlit
            st.pyplot(fig)

        if 'Precision Recall Curve' in metrics_list:
            st.subheader('precision Recall Curve')
            fig, ax = plt.subplots()
            # Plot PR curve using the model directly
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            # Create a matplotlib figure
            fig, ax = plt.subplots()
            # Plot ROC curve using sklearn
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

    st.sidebar.subheader('Choose Classififer')
    classifier = st.sidebar.selectbox('',('SVM','Logistic Regression','Random Forest'))

    if classifier == 'SVM':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C (Regularization paramter)', 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio('Kernel',('rbf','linear','sigmoid','poly'), key='kernel')
        gamma = st.sidebar.radio('Gamma', ('scale','auto'),key='gamma')
        metrics_list = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','Precision Recall Curve','ROC Curve'))

        if st.sidebar.button('Classify', key='classify'):
            st.subheader('SVM Results')
            # Build Model
            model = SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_train,y_train)   
            accuracy = model.score(X_test,y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy',  round(accuracy, 2))
            st.write('Precision', round(precision_score(y_test,y_pred,labels=class_names),2))
            st.write('Recall',round(recall_score(y_test,y_pred,labels=class_names),2))
            plot_metrics(metrics_list)

            
        
    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input('C (Regularization paramter)', 0.01, 10.0, step=0.01, key="C_LR")
        penalty = st.sidebar.radio('penalty',('l2'), key='penalty')
        max_iter = st.sidebar.slider("Maximum Number of Ierations", 100, 500, key='max_iter')
        metrics_list = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','Precision Recall Curve','ROC Curve'))   
        
        if st.sidebar.button('Classify', key='classify'):
                
            st.subheader('Logistic Regression Results')
                # Build Model
            model = LogisticRegression(C=C, penalty=penalty, max_iter=max_iter)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy',  round(accuracy, 2))
            st.write('Precision', round(precision_score(y_test,y_pred,labels=class_names),2))
            st.write('Recall',round(recall_score(y_test,y_pred,labels=class_names),2))
            plot_metrics(metrics_list)
            
    if classifier == 'Random Forest':
        st.sidebar.subheader('Model Hyperparameters')
        max_depth = st.sidebar.number_input('Max depth of trees', 100, 5000, step=10, key="max_depth")
        n_estimators = st.sidebar.number_input('Number of trees in the forest', 1, 20, step=1, key="n_estimators")
        max_features = st.sidebar.radio('Features to consider at each split',('sqrt','log2'), key='max_features')
        bootstrap = st.sidebar.radio('Bootstrap samples when building trees',(True,False), key='bootstrap')
        criterion = st.sidebar.radio('Function to measure quality of a split',('gini','entropy'), key='criterion')
        
        metrics_list = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix','Precision Recall Curve','ROC Curve'))   
        
        if st.sidebar.button('Classify', key='classify'):
                
            st.subheader('Random Forest Results')
                # Build Model
            model = RandomForestClassifier( max_depth=max_depth,
                                            n_estimators=n_estimators,
                                            max_features=max_features, 
                                            bootstrap=bootstrap,
                                            criterion=criterion,
                                            n_jobs=-1)
            model.fit(X_train,y_train)
            accuracy = model.score(X_test,y_test)
            y_pred = model.predict(X_test)
            st.write('Accuracy',  round(accuracy, 2))
            st.write('Precision', round(precision_score(y_test,y_pred,labels=class_names),2))
            st.write('Recall',round(recall_score(y_test,y_pred,labels=class_names),2))
            plot_metrics(metrics_list)

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://raw.githubusercontent.com/ahmedjajan93/Mushromms-Classification/main/img.png");
             background-size:cover;
             background-position:center;
             background-repeat: no-repeat;
             background-attachment: fiexed;
         }}
         </style>
         """,
           unsafe_allow_html=True
     )
            
        
if __name__ == '__main__':
    main()


