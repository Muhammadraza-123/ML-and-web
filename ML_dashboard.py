import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.header("Machine Learning Classifiers");
st.success("Dataset:");
df = pd.read_csv('music.csv');
st.write(df);
st.text("Here,age and gender is input features,genre is output features")
st.success("Dataset Graph");
st.bar_chart(df)
st.balloons();
st.sidebar.warning("Different classifiers selection-box");
code=st.sidebar.selectbox("select classifier",['any select','gnb','knn','clf','dt','rf']);
if code=="gnb":
    st.header("Naive Bayes Classifier Algorithm");
    with st.echo():
        import pandas as pd
        from sklearn.naive_bayes import GaussianNB
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        # Create a Gaussian Naive Bayes classifier
        gnb = GaussianNB()
        # Train the classifier on the data
        gnb.fit(X, y)
        # Make predictions on the data
        pred = gnb.predict([[28,1],[28,0]])
        # Print the prediction
        print("prediction : ",pred)
    st.error("Accuracy Code")
    with st.echo():
        import pandas as pd
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        #Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)
        # Create a Gaussian Naive Bayes classifier
        gnb = GaussianNB()
        # Train the classifier on the training data
        gnb.fit(X_train, y_train)
        # Make predictions on the testing data
        y_pred = gnb.predict(X_test)
        #calculate and print the Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy : ", accuracy)
        print("Actual : ",y_test)
        print("prediction : ",y_pred)
        
elif code=="knn":
    st.header("K-Nearest Neighbours Classifier Algorithm");
    with st.echo():
        import pandas as pd
        from sklearn.neighbors import KNeighborsClassifier
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        # Create a K-Nearest neighbors classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        # Train the classifier on the data
        knn.fit(X, y)
        # Make predictions on the data
        pred = knn.predict([[28,1],[28,0]])
        # Print the prediction
        print("prediction : ",pred)
    st.error("Accuracy Code")
    with st.echo():
        import pandas as pd
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        #Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)
        # Create a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        # Train the classifier on the training data
        knn.fit(X_train, y_train)
        # Make predictions on the testing data
        y_pred = knn.predict(X_test)
        #calculate and print the Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy : ", accuracy)
        print("Actual : ",y_test)
        print("prediction : ",y_pred)
        
elif code=="clf":
    st.header("Support Vector Machine Classifier Algorithm");
    with st.echo():
        import pandas as pd
        from sklearn.svm import SVC
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        # Create a Support Vector Machine classifier
        clf = SVC(kernel='linear')
        # Train the classifier on the data
        clf.fit(X, y)
        # Make predictions on the data
        pred = clf.predict([[28,1],[28,0]])
        # Print the prediction
        print("prediction : ",pred)
    st.error("Accuracy Code")
    with st.echo():
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        #Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)
        # Create a SVM classifier
        clf = SVC(kernel='linear')
        # Train the classifier on the training data
        clf.fit(X_train, y_train)
        # Make predictions on the testing data
        y_pred = clf.predict(X_test)
        #calculate and print the Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy : ", accuracy)
        print("Actual : ",y_test)
        print("prediction : ",y_pred)
        
elif code=="dt":
    st.header("Decision Tree Classifier Algorithm");
    with st.echo():
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        # Create a Decision Tree classifier
        dt = DecisionTreeClassifier()
        # Train the classifier on the data
        dt.fit(X, y)
        # Make predictions on the data
        pred = dt.predict([[28,1],[28,0]])
        # Print the prediction
        print("prediction : ",pred)
    st.error("Accuracy Code")
    with st.echo():    
        import pandas as pd
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        #Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)
        # Create a Decision Tree classifier
        dt = DecisionTreeClassifier()
        # Train the classifier on the training data
        dt.fit(X_train, y_train)
        # Make predictions on the testing data
        y_pred = dt.predict(X_test)
        #calculate and print the Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy : ", accuracy)
        print("Actual : ",y_test)
        print("prediction : ",y_pred)
        
elif code=="rf":
    st.header("Random Forest Classifier Algorithm");
    with st.echo():
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        # Create a Random Forest classifier
        rf = RandomForestClassifier(n_estimators=10)
        # Train the classifier on the data
        rf.fit(X, y)
        # Make predictions on the data
        pred = rf.predict([[28,1],[28,0]])
        # Print the prediction
        print("prediction : ",pred)
    st.error("Accuracy Code")
    with st.echo(): 
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # Load the dataset
        df = pd.read_csv('music.csv')
        # create input and output features from dataset
        X = df.drop(columns=['genre'])
        y = df['genre']
        #Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)
        # Create a Random Forest classifier
        rf =RandomForestClassifier(n_estimators=10)
        # Train the classifier on the training data
        rf.fit(X_train, y_train)
        # Make predictions on the testing data
        y_pred = rf.predict(X_test)
        #calculate and print the Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy : ", accuracy)
        print("Actual : ",y_test)
        print("prediction : ",y_pred)