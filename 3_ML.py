import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.title("TRAIN A REGRESSION MODEL")
st.write("# IMPLEMENT DIFFERENT MACHINE LEARNING ALGORITHM ON DIFFERENT DATASET")

upload_file = st.file_uploader("Upload CSV", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("### PREVIEW OF THE DATASET")
    st.dataframe(df.head(5))

    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    feature_cols = st.sidebar.multiselect(
        "Select Features Column",
        [col for col in df.columns if col != target_col]
    )

    if feature_cols:
        X = df[feature_cols].copy()
        y = df[target_col]

        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include='object').columns.tolist()

        #ENCODINg 
        if cat_cols:
            encoding = st.sidebar.selectbox(
                "Select Encoding Type",
                ["None", "OneHotEncoding", "LabelEncoding"] 
            )
            if encoding == "LabelEncoding":
                le = LabelEncoder()
                for col in cat_cols:
                    X[col] = le.fit_transform(X[col])
            elif encoding == "OneHotEncoding":
                X = pd.get_dummies(X, columns=cat_cols).astype(int)
                num_cols = X.select_dtypes(include=np.number).columns.tolist()  

        #IMPUTATION
        if num_cols:
            impute = st.sidebar.selectbox(
                "Handle Missing Values",
                ["None", "Fill with Mean", "Fill with Median"]
            )
            if impute == "Fill with Mean":
                X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
            elif impute == "Fill with Median":
                X[num_cols] = X[num_cols].fillna(X[num_cols].median())

        #SCALING 
        if num_cols:
            scaling = st.sidebar.selectbox(
                "Select Scaling Type",
                ["None", "StandardScaler", "MinMaxScaler"]
            )
            if scaling == "StandardScaler":
                scaler = StandardScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols]) 
            elif scaling == "MinMaxScaler":
                scaler = MinMaxScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols])  

        #MODEL & SPLIT SELECTION
        regression_name = st.sidebar.selectbox(        
            "Select Regressor",
            ["Linear Regression", "Decision Tree", "Random Forest Regressor", "XGBoost Regressor"]
        )

        division_name = st.sidebar.selectbox(           
            "Select Division Type",
            ["Train-Test-Split", "KFold"]
        )

        #HYPERPARAMETERS
        def parameter_ui(reg_name):
            params = {}
            if reg_name == "Decision Tree":
                params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 5)
                params['max_leaf_nodes'] = st.sidebar.slider("Max Leaf Nodes", 2, 50, 10)  
            elif reg_name == "Random Forest Regressor":
                params['n_estimators'] = st.sidebar.slider("N Estimators", 2, 100, 7)
                params['max_depth'] = st.sidebar.slider("Max Depth", 2, 10, 5)
            elif reg_name == "XGBoost Regressor":
                params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 5)
                params['n_estimators'] = st.sidebar.slider("N Estimators", 2, 100, 50)
                params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
            return params

        params = parameter_ui(regression_name)

        #GET REGRESSOR
        def get_regressor(reg_name, params):
            if reg_name == "XGBoost Regressor":           
                return XGBRegressor(
                    max_depth=params['max_depth'],
                    n_estimators=params['n_estimators'],
                    learning_rate=params['learning_rate']
                )
            elif reg_name == "Decision Tree":
                return DecisionTreeRegressor(
                    max_depth=params['max_depth'],
                    max_leaf_nodes=params['max_leaf_nodes']  
                )
            elif reg_name == "Random Forest Regressor":
                return RandomForestRegressor(
                    max_depth=params['max_depth'],
                    n_estimators=params['n_estimators'],
                    random_state=42                        
                )
            else:
                return LinearRegression()

        reg = get_regressor(regression_name, params)

        #PLOT
        def plot_actual_vs_predicted(y_test, y_pred):
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

        #TRAIN 
        if st.button("Train Model"):
            if division_name == "Train-Test-Split":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)

                st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
                st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                plot_actual_vs_predicted(y_test, y_pred)

            elif division_name == "KFold":
                n_split = st.sidebar.slider('n_splits',2,10,5)
                kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
                r2_scores, rmse_scores = [], []

                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    r2_scores.append(r2_score(y_test, y_pred))
                    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

                st.write(f"**Mean R² Score:** {np.mean(r2_scores):.4f}")
                st.write(f"**Mean RMSE:** {np.mean(rmse_scores):.4f}")
                plot_actual_vs_predicted(y_test, y_pred)  