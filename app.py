import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap


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


def get_tree_estimators(model):

    tree_estimators = []

    if hasattr(model, "named_estimators_"):

        for name, estimator in model.named_estimators_.items():

            if hasattr(estimator, "feature_importances_"):
                tree_estimators.append(
                    (name, estimator)
                )

    if not tree_estimators and hasattr(model, "estimators_"):

        for i, estimator in enumerate(model.estimators_):

            if hasattr(estimator, "feature_importances_"):
                tree_estimators.append(
                    (f"Estimator {i + 1}", estimator)
                )

    return tree_estimators


def get_feature_importance(model):

    if not hasattr(model, "feature_importances_"):
        return None, None

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    return type(model).__name__, importance_df


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
            value=20.0,
            step=1.0
        )

        ozone = st.number_input(
            "Ozone AQI",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with col2:

        no2 = st.number_input(
            "NO₂ AQI",
            min_value=0.0,
            value=30.0,
            step=1.0
        )

        pm25 = st.number_input(
            "PM2.5 AQI",
            min_value=0.0,
            value=100.0,
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

        predicted_aqi = max(
            0,
            predicted_aqi
        )

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
        "This section describes the ensemble models and their "
        "feature importance."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Regression Model")

        st.write(
            f"**Model:** `{type(regression_model).__name__}`"
        )

        regression_estimators = get_ensemble_estimators(
            regression_model
        )

        if regression_estimators:

            st.write("**Ensemble Components:**")

            for name, estimator in regression_estimators.items():

                st.write(
                    f"- `{name}` — {type(estimator).__name__}"
                )

        else:

            st.write(
                "No named ensemble components found."
            )

    with col2:

        st.subheader("Classification Model")

        st.write(
            f"**Model:** `{type(classification_model).__name__}`"
        )

        classification_estimators = get_ensemble_estimators(
            classification_model
        )

        if classification_estimators:

            st.write("**Ensemble Components:**")

            for name, estimator in classification_estimators.items():

                st.write(
                    f"- `{name}` — {type(estimator).__name__}"
                )

        else:

            st.write(
                "No named ensemble components found."
            )

    st.divider()

    st.subheader("Saved Regression Model Structure")

    st.write(
        "Regression model:",
        type(regression_model).__name__
    )

    st.write(
        "Has named estimators:",
        hasattr(
            regression_model,
            "named_estimators_"
        )
    )

    st.write(
        "Has estimators:",
        hasattr(
            regression_model,
            "estimators_"
        )
    )

    if hasattr(
        regression_model,
        "named_estimators_"
    ):

        st.write(
            "Estimators:",
            list(
                regression_model.named_estimators_.keys()
            )
        )

    st.divider()

    st.subheader("Regression Feature Importance")

    reg_name, reg_importance = get_feature_importance(
        regression_model
    )

    if reg_importance is not None:

        st.write(
            f"Tree-based components used: **{reg_name}**"
        )

        st.bar_chart(
            reg_importance.set_index(
                "Feature"
            )["Importance"]
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
            f"Tree-based components used: **{clf_name}**"
        )

        st.bar_chart(
            clf_importance.set_index(
                "Feature"
            )["Importance"]
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
        "Feature importance indicates how the tree-based models "
        "use the pollutant features during prediction. It does "
        "not represent a causal relationship."
    )


with tab3:

    st.header("🔎 Explainability")

    st.write(
        "Understand how each pollutant contributes to the "
        "predicted AQI using SHAP."
    )

    st.info(
        "SHAP values explain the model's prediction. "
        "They do not represent causal effects."
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
            key="shap_co"
        )

        explain_ozone = st.number_input(
            "Ozone AQI",
            min_value=0.0,
            value=50.0,
            step=1.0,
            key="shap_ozone"
        )

    with col2:

        explain_no2 = st.number_input(
            "NO₂ AQI",
            min_value=0.0,
            value=30.0,
            step=1.0,
            key="shap_no2"
        )

        explain_pm25 = st.number_input(
            "PM2.5 AQI",
            min_value=0.0,
            value=100.0,
            step=1.0,
            key="shap_pm25"
        )

    explain_input = create_input(
        explain_co,
        explain_ozone,
        explain_no2,
        explain_pm25
    )

    st.divider()

    if st.button(
        "🔍 Explain Prediction",
        type="primary",
        use_container_width=True
    ):

        predicted_aqi = regression_model.predict(
            explain_input
        )[0]

        predicted_aqi = max(
            0,
            predicted_aqi
        )

        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.2f}"
        )

        st.divider()

        try:

            st.subheader(
                "SHAP Feature Contributions"
            )

            background = pd.DataFrame({
                "CO AQI Value": [
                    explain_co * 0.5,
                    explain_co,
                    explain_co * 1.5
                ],
                "Ozone AQI Value": [
                    explain_ozone * 0.5,
                    explain_ozone,
                    explain_ozone * 1.5
                ],
                "NO2 AQI Value": [
                    explain_no2 * 0.5,
                    explain_no2,
                    explain_no2 * 1.5
                ],
                "PM2.5 AQI Value": [
                    explain_pm25 * 0.5,
                    explain_pm25,
                    explain_pm25 * 1.5
                ]
            })

            background = background.replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

            explainer = shap.Explainer(
                regression_model.predict,
                background,
                feature_names=FEATURES
            )

            shap_result = explainer(
                explain_input
            )

            shap_values = np.asarray(
                shap_result.values
            )

            if shap_values.ndim == 3:

                shap_values = shap_values[0, :, 0]

            elif shap_values.ndim == 2:

                shap_values = shap_values[0]

            else:

                shap_values = shap_values.flatten()

            shap_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP Value": shap_values
            })

            shap_df["Absolute Impact"] = np.abs(
                shap_df["SHAP Value"]
            )

            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            )

            st.bar_chart(
                shap_df.set_index(
                    "Feature"
                )["SHAP Value"]
            )

            st.divider()

            st.subheader(
                "Detailed Explanation"
            )

            display_df = shap_df.copy()

            display_df["Direction"] = display_df[
                "SHAP Value"
            ].apply(
                lambda x:
                "Increases AQI"
                if x > 0
                else "Decreases AQI"
                if x < 0
                else "No effect"
            )

            st.dataframe(
                display_df[
                    [
                        "Feature",
                        "SHAP Value",
                        "Absolute Impact",
                        "Direction"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            most_important = shap_df.iloc[0]

            st.success(
                f"Most influential feature for this prediction: "
                f"{most_important['Feature']} "
                f"({most_important['SHAP Value']:+.3f})"
            )

            positive_features = shap_df[
                shap_df["SHAP Value"] > 0
            ]

            negative_features = shap_df[
                shap_df["SHAP Value"] < 0
            ]

            if len(positive_features) > 0:

                st.write(
                    "🟢 Features pushing the prediction higher:"
                )

                for _, row in positive_features.iterrows():

                    st.write(
                        f"- {row['Feature']}: "
                        f"+{row['SHAP Value']:.3f}"
                    )

            if len(negative_features) > 0:

                st.write(
                    "🔵 Features pushing the prediction lower:"
                )

                for _, row in negative_features.iterrows():

                    st.write(
                        f"- {row['Feature']}: "
                        f"{row['SHAP Value']:.3f}"
                    )

        except Exception as e:

            st.error(
                "SHAP explanation could not be generated."
            )

            st.code(
                str(e)
            )


with tab4:

    st.header("🧪 What-If Analysis")

    st.write(
        "Change one pollutant value while keeping the other "
        "values fixed and observe how the model prediction changes."
    )

    st.warning(
        "This is a model-based hypothetical analysis. "
        "It does not establish a causal relationship."
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

        st.subheader("AQI Comparison")

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
            comparison.set_index(
                "Scenario"
            )
        )

        st.subheader("Input Comparison")

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
