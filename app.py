import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt




st.set_page_config(
    page_title="Explainable AQI Estimation",
    page_icon="🌍",
    layout="wide"
)




@st.cache_resource
def load_models():
    regression_model = joblib.load("aqi_regression_model.pkl")
    classification_model = joblib.load("aqi_classification_model.pkl")
    return regression_model, classification_model


regression_model, classification_model = load_models()




FEATURES = [
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value"
]

DEFAULT_VALUES = {
    "CO AQI Value": 20.0,
    "Ozone AQI Value": 50.0,
    "NO2 AQI Value": 30.0,
    "PM2.5 AQI Value": 100.0
}



def get_tree_estimator(model):
    """
    Find a tree-based estimator inside a Voting ensemble.
    """

    if hasattr(model, "named_estimators_"):

        estimators = model.named_estimators_

        preferred = [
            "extra_trees",
            "extratrees",
            "random_forest",
            "rf",
            "randomforest",
            "gradient_boosting",
            "gb"
        ]

        for name in preferred:
            if name in estimators:
                estimator = estimators[name]

                if hasattr(estimator, "feature_importances_"):
                    return name, estimator

        for name, estimator in estimators.items():

            if hasattr(estimator, "feature_importances_"):
                return name, estimator

    return None, None


def get_feature_importance(model):

    name, estimator = get_tree_estimator(model)

    if estimator is None:
        return None, None

    importance = estimator.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=False
    )

    return name, importance_df


def create_input(co, ozone, no2, pm25):

    return pd.DataFrame({
        "CO AQI Value": [co],
        "Ozone AQI Value": [ozone],
        "NO2 AQI Value": [no2],
        "PM2.5 AQI Value": [pm25]
    })


def get_ensemble_estimators(model):

    if hasattr(model, "named_estimators_"):
        return model.named_estimators_

    return {}


def model_name(model):
    return type(model).__name__




st.title("🌍 Explainable AQI Estimation")

st.write(
    "Estimate Air Quality Index using pollutant-level AQI values "
    "with ensemble machine learning and interpretable model analysis."
)

st.caption(
    "Inputs: CO, Ozone, NO₂ and PM2.5 AQI values"
)

st.divider()




tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predict AQI",
    "📊 Model Analysis",
    "🔎 Explainability",
    "🧪 What-If Analysis"
])




with tab1:

    st.header("Predict Air Quality")

    st.write(
        "Enter pollutant-level AQI values to estimate the overall AQI "
        "and its corresponding category."
    )

    col1, col2 = st.columns(2)

    with col1:

        co = st.number_input(
            "CO AQI",
            min_value=0.0,
            value=DEFAULT_VALUES["CO AQI Value"],
            step=1.0
        )

        ozone = st.number_input(
            "Ozone AQI",
            min_value=0.0,
            value=DEFAULT_VALUES["Ozone AQI Value"],
            step=1.0
        )

    with col2:

        no2 = st.number_input(
            "NO₂ AQI",
            min_value=0.0,
            value=DEFAULT_VALUES["NO2 AQI Value"],
            step=1.0
        )

        pm25 = st.number_input(
            "PM2.5 AQI",
            min_value=0.0,
            value=DEFAULT_VALUES["PM2.5 AQI Value"],
            step=1.0
        )

    input_data = create_input(
        co,
        ozone,
        no2,
        pm25
    )

    st.divider()

    if st.button(
        "Predict AQI",
        type="primary",
        use_container_width=True
    ):

        predicted_aqi = regression_model.predict(
            input_data
        )[0]

        predicted_category = classification_model.predict(
            input_data
        )[0]

        predicted_aqi = max(0, predicted_aqi)

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted AQI",
                f"{predicted_aqi:.2f}"
            )

        with col2:

            st.metric(
                "AQI Category",
                str(predicted_category)
            )

        st.divider()

        st.subheader("Input Pollutant Values")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "The prediction represents the model's estimated AQI "
            "based on the entered pollutant AQI values."
        )




with tab2:

    st.header("📊 Model Analysis")

    st.write(
        "This section describes the ensemble models used for AQI "
        "regression and AQI category classification."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Regression Model")

        st.write(
            f"**Final model:** `{model_name(regression_model)}`"
        )

        regression_estimators = get_ensemble_estimators(
            regression_model
        )

        if regression_estimators:

            st.write("**Ensemble components:**")

            for name, estimator in regression_estimators.items():

                st.write(
                    f"- `{name}` — {type(estimator).__name__}"
                )

        else:

            st.write(
                "The saved model does not expose named ensemble components."
            )

    with col2:

        st.subheader("Classification Model")

        st.write(
            f"**Final model:** `{model_name(classification_model)}`"
        )

        classification_estimators = get_ensemble_estimators(
            classification_model
        )

        if classification_estimators:

            st.write("**Ensemble components:**")

            for name, estimator in classification_estimators.items():

                st.write(
                    f"- `{name}` — {type(estimator).__name__}"
                )

        else:

            st.write(
                "The saved model does not expose named ensemble components."
            )

    st.divider()


    st.subheader("Regression Feature Importance")

    reg_name, reg_importance = get_feature_importance(
        regression_model
    )

    if reg_importance is not None:

        st.write(
            f"Feature importance from the tree-based "
            f"ensemble component: **{reg_name}**"
        )

        st.bar_chart(
            reg_importance.set_index("Feature")["Importance"]
        )

        st.dataframe(
            reg_importance,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Tree-based feature importance is not available "
            "for the saved regression model."
        )

    st.divider()

    

    st.subheader("Classification Feature Importance")

    clf_name, clf_importance = get_feature_importance(
        classification_model
    )

    if clf_importance is not None:

        st.write(
            f"Feature importance from the tree-based "
            f"ensemble component: **{clf_name}**"
        )

        st.bar_chart(
            clf_importance.set_index("Feature")["Importance"]
        )

        st.dataframe(
            clf_importance,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Tree-based feature importance is not available "
            "for the saved classification model."
        )

    st.divider()

    st.info(
        "Feature importance indicates how much the selected "
        "tree-based model component uses each pollutant feature "
        "when making predictions. It should not be interpreted "
        "as a causal effect."
    )


with tab3:

    st.header("🔎 Explainability")

    st.write(
        "Use SHAP to inspect how individual pollutant inputs "
        "contribute to a model prediction."
    )

    st.warning(
        "SHAP explanations shown here are model explanations, "
        "not causal relationships between pollutants and AQI."
    )

    st.divider()



    st.subheader("Enter Values to Explain")

    col1, col2 = st.columns(2)

    with col1:

        explain_co = st.number_input(
            "CO AQI",
            min_value=0.0,
            value=20.0,
            step=1.0,
            key="explain_co"
        )

        explain_ozone = st.number_input(
            "Ozone AQI",
            min_value=0.0,
            value=50.0,
            step=1.0,
            key="explain_ozone"
        )

    with col2:

        explain_no2 = st.number_input(
            "NO₂ AQI",
            min_value=0.0,
            value=30.0,
            step=1.0,
            key="explain_no2"
        )

        explain_pm25 = st.number_input(
            "PM2.5 AQI",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key="explain_pm25"
        )

    explain_input = create_input(
        explain_co,
        explain_ozone,
        explain_no2,
        explain_pm25
    )

    st.divider()

   

    if st.button(
        "Explain Prediction",
        type="primary",
        use_container_width=True
    ):

        tree_name, tree_model = get_tree_estimator(
            regression_model
        )

        if tree_model is None:

            st.error(
                "A tree-based estimator could not be found "
                "inside the saved regression ensemble."
            )

        else:

            try:

                explainer = shap.TreeExplainer(
                    tree_model
                )

                shap_values = explainer(
                    explain_input
                )

                values = shap_values.values[0]

                if values.ndim > 1:
                    values = values[:, 0]

                explanation_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "SHAP Value": values,
                    "Absolute Impact": np.abs(values)
                }).sort_values(
                    "Absolute Impact",
                    ascending=False
                )

                st.subheader(
                    f"SHAP Explanation — {tree_name}"
                )

                st.write(
                    "Positive SHAP values push the model prediction "
                    "higher, while negative values push it lower."
                )

                st.bar_chart(
                    explanation_df.set_index(
                        "Feature"
                    )["SHAP Value"]
                )

                st.dataframe(
                    explanation_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()

                st.subheader("Prediction Being Explained")

                explained_prediction = regression_model.predict(
                    explain_input
                )[0]

                st.metric(
                    "Ensemble AQI Prediction",
                    f"{max(0, explained_prediction):.2f}"
                )

                st.caption(
                    "The SHAP values above explain the selected "
                    "tree-based component of the ensemble. The AQI "
                    "shown here is the final ensemble prediction."
                )

            except Exception as e:

                st.error(
                    "SHAP explanation could not be generated."
                )

                st.code(str(e))



with tab4:

    st.header("🧪 What-If Analysis")

    st.write(
        "Change one pollutant value while keeping the other "
        "values fixed and observe how the model prediction changes."
    )

    st.warning(
        "This is a model-based hypothetical analysis. "
        "It does not establish that changing a pollutant would "
        "causally change real-world AQI by the displayed amount."
    )

    st.divider()

    st.subheader("Baseline Values")

    col1, col2 = st.columns(2)

    with col1:

        base_co = st.number_input(
            "Baseline CO AQI",
            min_value=0.0,
            value=20.0,
            step=1.0,
            key="base_co"
        )

        base_ozone = st.number_input(
            "Baseline Ozone AQI",
            min_value=0.0,
            value=50.0,
            step=1.0,
            key="base_ozone"
        )

    with col2:

        base_no2 = st.number_input(
            "Baseline NO₂ AQI",
            min_value=0.0,
            value=30.0,
            step=1.0,
            key="base_no2"
        )

        base_pm25 = st.number_input(
            "Baseline PM2.5 AQI",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key="base_pm25"
        )

    baseline = create_input(
        base_co,
        base_ozone,
        base_no2,
        base_pm25
    )

    baseline_prediction = regression_model.predict(
        baseline
    )[0]

    baseline_prediction = max(
        0,
        baseline_prediction
    )

    st.metric(
        "Baseline Predicted AQI",
        f"{baseline_prediction:.2f}"
    )

    st.divider()

    st.subheader("Modify One Pollutant")

    pollutant = st.selectbox(
        "Select pollutant",
        FEATURES
    )

    new_value = st.number_input(
        "New AQI value",
        min_value=0.0,
        value=float(
            baseline[pollutant].iloc[0]
        ),
        step=1.0
    )

    modified = baseline.copy()

    modified[pollutant] = new_value

    if st.button(
        "Run What-If Analysis",
        type="primary",
        use_container_width=True
    ):

        new_prediction = regression_model.predict(
            modified
        )[0]

        new_prediction = max(
            0,
            new_prediction
        )

        difference = (
            new_prediction -
            baseline_prediction
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Baseline AQI",
                f"{baseline_prediction:.2f}"
            )

        with col2:

            st.metric(
                "What-If AQI",
                f"{new_prediction:.2f}"
            )

        with col3:

            st.metric(
                "Change",
                f"{difference:+.2f}"
            )

        st.divider()

        st.subheader("Comparison")

        comparison = pd.DataFrame({
            "Scenario": [
                "Baseline",
                "What-If"
            ],
            "AQI": [
                baseline_prediction,
                new_prediction
            ]
        })

        st.bar_chart(
            comparison.set_index("Scenario")
        )

        st.subheader("Changed Input")

        changed_values = pd.DataFrame({
            "Feature": FEATURES,
            "Baseline": [
                baseline[f].iloc[0]
                for f in FEATURES
            ],
            "What-If": [
                modified[f].iloc[0]
                for f in FEATURES
            ]
        })

        changed_values["Change"] = (
            changed_values["What-If"] -
            changed_values["Baseline"]
        )

        st.dataframe(
            changed_values,
            use_container_width=True,
            hide_index=True
        )




st.divider()

st.caption(
    "Explainable AQI Estimation using Ensemble Machine Learning"
)

st.caption(
    "Models: VotingRegressor + VotingClassifier | "
    "Features: CO, Ozone, NO₂ and PM2.5 AQI values"
)
