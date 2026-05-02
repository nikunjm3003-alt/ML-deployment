import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

'''IN THIS I HAVE NOT PUT SCALING AND ENCODING METHOD SO I AM ONLY PUTTING DATASET WITH NUMERIC VALUE. I HAVE ALSO 
 MENTIONED THE DATASET FOR CONVINENCE'''


# title
st.title("A REGRESSION MODEL")
st.write("# TRYING DIFFERENT TYPE OF MODEL ON DIFFERENT DATASETS")

# ---- FILE UPLOAD ----
upload_file = st.file_uploader("Upload CSV", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Let user pick target and feature columns
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)
    feature_cols = st.sidebar.multiselect("Select Feature Columns", df.columns)

    if feature_cols:
        X = df[feature_cols].values
        y = df[target_col].values

        # ---- SIDEBAR ----
        regressor_name = st.sidebar.selectbox("Select Regressor", (
            "XGBOOST Regressor", "Decision Tree Regressor", "Random Forest Regressor"))

        division_name = st.sidebar.selectbox("How would you like to divide your data", (
            "Train-Test-Split", "K-Fold"))

        # ---- PARAMETERS ----
        def parameter_ui(reg_name):
            params = {}
            if reg_name == "XGBOOST Regressor":
                params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 5)
                params['n_estimators'] = st.sidebar.slider("N Estimators", 2, 100, 50)
                params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)

            elif reg_name == "Decision Tree Regressor":
                params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 5)
                params['max_leaf_nodes'] = st.sidebar.slider("Max Leaf Nodes", 2, 50, 10)

            else:
                params['n_estimators'] = st.sidebar.slider("N Estimators", 2, 150, 7)
                params['max_depth'] = st.sidebar.slider("Max Depth", 2, 10, 5)
            return params

        params = parameter_ui(regressor_name)

        # ---- GET REGRESSOR ----
        def get_regressor(reg_name, params):
            if reg_name == 'XGBOOST Regressor':
                reg = XGBRegressor(
                    max_depth=params['max_depth'],
                    n_estimators=params['n_estimators'],
                    learning_rate=params['learning_rate']
                )
            elif reg_name == "Decision Tree Regressor":
                reg = DecisionTreeRegressor(
                    max_depth=params['max_depth'],
                    max_leaf_nodes=params['max_leaf_nodes']
                )
            else:
                reg = RandomForestRegressor(
                    max_depth=params['max_depth'],
                    n_estimators=params['n_estimators']
                )
            return reg

        reg = get_regressor(regressor_name, params)

        def plot_actual_vs_predicted(y_test, y_pred):
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)


        # ---- TRAIN ACCORDING TO USER CHOICE ----
        if division_name == "Train-Test-Split":
            test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            st.write("## Results (Train-Test-Split)")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
            st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            st.write("### Actual vs Predicted")
            plot_actual_vs_predicted(y_test, y_pred)

        elif division_name == "K-Fold":
            k = st.sidebar.slider("Number of Folds (K)", 2, 10, 5)
            kf = KFold(n_splits=k, shuffle=True, random_state=42)

            # cross_val_score handles fitting automatically
            r2_scores = cross_val_score(reg, X, y, cv=kf, scoring='r2')
            mse_scores = cross_val_score(reg, X, y, cv=kf, scoring='neg_mean_squared_error')

            st.write("## Results (K-Fold)")
            st.write(f"**Mean R² Score:** {r2_scores.mean():.2f}")
            st.write(f"**Std R² Score:** {r2_scores.std():.2f}")
            st.write(f"**Mean MSE:** {(-mse_scores).mean():.2f}")
            st.write("### Actual vs Predicted")
            st.write("### R² Score per Fold")
            fig, ax = plt.subplots()
            ax.bar(range(1, k+1), r2_scores, color='steelblue', edgecolor='black')
            ax.axhline(y=r2_scores.mean(), color='r', linestyle='--', label='Mean R²')
            ax.set_xlabel("Fold")
            ax.set_ylabel("R² Score")
            ax.set_title("R² Score per Fold")
            ax.legend()
            st.pyplot(fig)

    else:
        st.warning("Please select at least one feature column from the sidebar")

else:
    st.info("Please upload a CSV file to get started")