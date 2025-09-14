import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import io
import base64
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Suicide Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Suicide Data Analysis Dashboard")
st.markdown("""
This interactive dashboard analyzes suicide statistics by demographic factors including age, sex, and race.
Upload your CSV data to explore patterns, perform clustering analysis, and build predictive models.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_option = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Data Upload & Overview", "Exploratory Analysis", "Clustering Analysis", "Classification Model", "Insights & Summary"]
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

def load_sample_data():
    """Load sample data for demonstration"""
    data = [
        ("10-14", "Male", "White", 300, 10000000),
        ("10-14", "Female", "White", 150, 10000000),
        ("15-24", "Male", "White", 4500, 20000000),
        ("15-24", "Female", "White", 1800, 20000000),
        ("25-34", "Male", "White", 4000, 12000000),
        ("25-34", "Female", "White", 1200, 11000000),
        ("35-44", "Male", "White", 5200, 10000000),
        ("35-44", "Female", "White", 1500, 10000000),
        ("45-54", "Male", "White", 7800, 11000000),
        ("45-54", "Female", "White", 2100, 11000000),
        ("55-64", "Male", "White", 9200, 12000000),
        ("55-64", "Female", "White", 2800, 12000000),
        ("65+", "Male", "White", 15000, 18000000),
        ("65+", "Female", "White", 3000, 20000000),
        ("15-24", "Male", "Black", 1200, 3000000),
        ("15-24", "Female", "Black", 400, 3200000),
        ("25-34", "Male", "Black", 1800, 2800000),
        ("25-34", "Female", "Black", 450, 2900000),
        ("35-44", "Male", "Black", 2100, 2500000),
        ("35-44", "Female", "Black", 520, 2600000),
        ("45-54", "Male", "Black", 2800, 2700000),
        ("45-54", "Female", "Black", 680, 2800000),
        ("55-64", "Male", "Black", 3200, 2900000),
        ("55-64", "Female", "Black", 720, 3000000),
        ("65+", "Male", "Black", 1200, 5000000),
        ("65+", "Female", "Black", 800, 6000000),
        ("15-24", "Male", "Hispanic", 900, 4000000),
        ("15-24", "Female", "Hispanic", 280, 4200000),
        ("25-34", "Male", "Hispanic", 1400, 3800000),
        ("25-34", "Female", "Hispanic", 350, 3900000),
        ("35-44", "Male", "Hispanic", 1600, 3500000),
        ("35-44", "Female", "Hispanic", 380, 3600000),
        ("45-54", "Male", "Hispanic", 2100, 3200000),
        ("45-54", "Female", "Hispanic", 480, 3300000),
        ("55-64", "Male", "Hispanic", 2400, 2800000),
        ("55-64", "Female", "Hispanic", 520, 2900000),
        ("65+", "Male", "Hispanic", 900, 4000000),
        ("65+", "Female", "Hispanic", 500, 4200000),
    ]
    
    df = pd.DataFrame(data, columns=["AgeGroup", "Sex", "Race", "Suicides", "Population"])
    return df

def calculate_rates(df):
    """Calculate suicide rates per 100,000 population"""
    df["Rate_per_100k"] = (df["Suicides"] / df["Population"]) * 100000
    return df

def encode_categorical_features(df):
    """Encode categorical features for machine learning"""
    le_age, le_sex, le_race = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df["Age_encoded"] = le_age.fit_transform(df["AgeGroup"])
    df["Sex_encoded"] = le_sex.fit_transform(df["Sex"])
    df["Race_encoded"] = le_race.fit_transform(df["Race"])
    return df, le_age, le_sex, le_race

def create_download_link(df, filename, file_format="csv"):
    """Create download link for DataFrame"""
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}.csv</a>'
    elif file_format == "json":
        json_str = df.to_json(orient='records', indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download {filename}.json</a>'
    return href

def export_analysis_results(data_dict, analysis_type):
    """Export analysis results as JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{analysis_type}_analysis_{timestamp}"
    
    json_str = json.dumps(data_dict, indent=2, default=str)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}.json">Download Analysis Results</a>'
    return href

# Data Upload & Overview Section
if analysis_option == "Data Upload & Overview":
    st.header("📁 Data Upload & Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with columns: AgeGroup, Sex, Race, Suicides, Population",
            type="csv",
            help="Your CSV should contain columns for demographic data and suicide statistics"
        )
    
    with col2:
        st.subheader("Or Use Sample Data")
        if st.button("Load Sample Dataset", type="primary"):
            st.session_state.df = load_sample_data()
            st.session_state.df = calculate_rates(st.session_state.df)
            st.session_state.data_loaded = True
            st.success("Sample data loaded successfully!")
            st.rerun()
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ["AgeGroup", "Sex", "Race", "Suicides", "Population"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.info("Please ensure your CSV contains: AgeGroup, Sex, Race, Suicides, Population")
            else:
                st.session_state.df = calculate_rates(df)
                st.session_state.data_loaded = True
                st.success("Data uploaded and processed successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Display data overview if loaded
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.subheader("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.df))
        with col2:
            st.metric("Age Groups", st.session_state.df["AgeGroup"].nunique())
        with col3:
            st.metric("Races", st.session_state.df["Race"].nunique())
        with col4:
            st.metric("Average Rate", f"{st.session_state.df['Rate_per_100k'].mean():.1f}")
        
        st.subheader("📊 Data Sample")
        st.dataframe(st.session_state.df, use_container_width=True)
        
        st.subheader("📈 Basic Statistics")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        # Export options
        st.subheader("📤 Export Data")
        col1, col2 = st.columns(2)
        with col1:
            # Raw data export
            st.markdown(create_download_link(st.session_state.df, "suicide_data", "csv"), unsafe_allow_html=True)
        with col2:
            # Statistics export
            stats_df = st.session_state.df.describe()
            st.markdown(create_download_link(stats_df, "data_statistics", "csv"), unsafe_allow_html=True)

# Exploratory Analysis Section
elif analysis_option == "Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the 'Data Upload & Overview' section.")
    else:
        df = st.session_state.df
        
        # Age Group Analysis
        st.subheader("📊 Suicide Rates by Age Group")
        age_rates = df.groupby("AgeGroup")["Rate_per_100k"].mean().reset_index()
        fig_age = px.bar(
            age_rates, 
            x="AgeGroup", 
            y="Rate_per_100k",
            title="Average Suicide Rate by Age Group",
            color="Rate_per_100k",
            color_continuous_scale="Reds"
        )
        fig_age.update_layout(xaxis_title="Age Group", yaxis_title="Rate per 100,000")
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Sex Analysis
        st.subheader("👥 Suicide Rates by Sex")
        col1, col2 = st.columns(2)
        
        with col1:
            sex_rates = df.groupby("Sex")["Rate_per_100k"].mean().reset_index()
            fig_sex = px.pie(
                sex_rates, 
                values="Rate_per_100k", 
                names="Sex",
                title="Suicide Rates by Sex"
            )
            st.plotly_chart(fig_sex, use_container_width=True)
        
        with col2:
            fig_sex_bar = px.bar(
                sex_rates, 
                x="Sex", 
                y="Rate_per_100k",
                title="Average Suicide Rate by Sex",
                color="Sex"
            )
            st.plotly_chart(fig_sex_bar, use_container_width=True)
        
        # Race Analysis
        st.subheader("🌍 Suicide Rates by Race")
        race_rates = df.groupby("Race")["Rate_per_100k"].mean().reset_index()
        fig_race = px.bar(
            race_rates, 
            x="Race", 
            y="Rate_per_100k",
            title="Average Suicide Rate by Race",
            color="Race"
        )
        st.plotly_chart(fig_race, use_container_width=True)
        
        # Heatmap Analysis
        st.subheader("🔥 Heatmap: Rates by Age Group and Sex")
        pivot_data = df.pivot_table(
            values="Rate_per_100k", 
            index="AgeGroup", 
            columns="Sex", 
            aggfunc="mean"
        )
        
        fig_heatmap = px.imshow(
            pivot_data,
            title="Suicide Rates Heatmap (Age Group vs Sex)",
            color_continuous_scale="Reds",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Export exploratory analysis results
        st.subheader("📤 Export Analysis Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Age group analysis
            age_rates = df.groupby("AgeGroup")["Rate_per_100k"].mean().reset_index()
            st.markdown(create_download_link(age_rates, "age_group_analysis", "csv"), unsafe_allow_html=True)
        
        with col2:
            # Sex analysis  
            sex_rates = df.groupby("Sex")["Rate_per_100k"].mean().reset_index()
            st.markdown(create_download_link(sex_rates, "sex_analysis", "csv"), unsafe_allow_html=True)
        
        with col3:
            # Race analysis
            race_rates = df.groupby("Race")["Rate_per_100k"].mean().reset_index()
            st.markdown(create_download_link(race_rates, "race_analysis", "csv"), unsafe_allow_html=True)

# Clustering Analysis Section
elif analysis_option == "Clustering Analysis":
    st.header("🎯 Clustering Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the 'Data Upload & Overview' section.")
    else:
        df = st.session_state.df.copy()
        df_encoded, le_age, le_sex, le_race = encode_categorical_features(df)
        
        st.subheader("⚙️ Clustering Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=3)
        with col2:
            features_to_use = st.multiselect(
                "Features for Clustering",
                ["Age_encoded", "Sex_encoded", "Race_encoded", "Rate_per_100k"],
                default=["Age_encoded", "Sex_encoded", "Race_encoded", "Rate_per_100k"]
            )
        
        if st.button("Run Clustering Analysis", type="primary"):
            if len(features_to_use) < 2:
                st.error("Please select at least 2 features for clustering.")
            else:
                # Prepare features
                X = df_encoded[features_to_use]
                
                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_encoded["Cluster"] = kmeans.fit_predict(X)
                
                # Display results
                st.subheader("🎯 Clustering Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Cluster Assignments:**")
                    cluster_summary = df_encoded.groupby("Cluster").agg({
                        "Rate_per_100k": ["mean", "count"],
                        "AgeGroup": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Mixed",
                        "Sex": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Mixed",
                        "Race": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Mixed"
                    }).round(2)
                    
                    cluster_summary.columns = ["Avg_Rate", "Count", "Primary_Age", "Primary_Sex", "Primary_Race"]
                    st.dataframe(cluster_summary, use_container_width=True)
                
                with col2:
                    # Cluster distribution pie chart
                    cluster_counts = df_encoded["Cluster"].value_counts().reset_index()
                    cluster_counts.columns = ["Cluster", "Count"]
                    cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)
                    
                    fig_clusters = px.pie(
                        cluster_counts,
                        values="Count",
                        names="Cluster",
                        title="Cluster Distribution"
                    )
                    st.plotly_chart(fig_clusters, use_container_width=True)
                
                # Scatter plot visualization
                st.subheader("📈 Cluster Visualization")
                if len(features_to_use) >= 2:
                    fig_scatter = px.scatter(
                        df_encoded,
                        x=features_to_use[0],
                        y=features_to_use[1] if len(features_to_use) > 1 else features_to_use[0],
                        color="Cluster",
                        size="Rate_per_100k",
                        hover_data=["AgeGroup", "Sex", "Race", "Rate_per_100k"],
                        title=f"Clusters by {features_to_use[0]} vs {features_to_use[1] if len(features_to_use) > 1 else features_to_use[0]}"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Detailed cluster analysis
                st.subheader("📋 Detailed Cluster Analysis")
                for cluster_id in sorted(df_encoded["Cluster"].unique()):
                    with st.expander(f"Cluster {cluster_id} Details"):
                        cluster_data = df_encoded[df_encoded["Cluster"] == cluster_id]
                        st.write(f"**Size:** {len(cluster_data)} records")
                        st.write(f"**Average Rate:** {cluster_data['Rate_per_100k'].mean():.2f} per 100k")
                        st.write("**Sample Records:**")
                        st.dataframe(
                            cluster_data[["AgeGroup", "Sex", "Race", "Rate_per_100k"]].head(),
                            use_container_width=True
                        )
                
                # Export clustering results
                st.subheader("📤 Export Clustering Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cluster summary export
                    st.markdown(create_download_link(cluster_summary, "cluster_summary", "csv"), unsafe_allow_html=True)
                
                with col2:
                    # Full clustered data export
                    clustered_data = df_encoded[["AgeGroup", "Sex", "Race", "Rate_per_100k", "Cluster"]]
                    st.markdown(create_download_link(clustered_data, "clustered_data", "csv"), unsafe_allow_html=True)

# Classification Model Section
elif analysis_option == "Classification Model":
    st.header("🤖 Classification Model")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the 'Data Upload & Overview' section.")
    else:
        df = st.session_state.df.copy()
        df_encoded, le_age, le_sex, le_race = encode_categorical_features(df)
        
        st.subheader("⚙️ Model Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_threshold = st.slider(
                "Risk Threshold (percentile)", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5, 
                step=0.1,
                help="Data points above this percentile will be labeled as 'High Risk'"
            )
        
        with col2:
            selected_models = st.multiselect(
                "Select Models to Compare",
                ["Decision Tree", "Random Forest", "SVM"],
                default=["Decision Tree", "Random Forest", "SVM"]
            )
        
        with col3:
            max_depth = st.slider("Maximum Tree Depth", min_value=2, max_value=10, value=3)
        
        if st.button("Train Classification Models", type="primary"):
            if not selected_models:
                st.error("Please select at least one model to train.")
            else:
                # Create high/low risk labels
                threshold_value = df_encoded["Rate_per_100k"].quantile(risk_threshold)
                df_encoded["HighRisk"] = (df_encoded["Rate_per_100k"] > threshold_value).astype(int)
                
                # Prepare features and target
                features = ["Age_encoded", "Sex_encoded", "Race_encoded"]
                X_class = df_encoded[features]
                y_class = df_encoded["HighRisk"]
                
                # Initialize models
                models = {}
                if "Decision Tree" in selected_models:
                    models["Decision Tree"] = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                if "Random Forest" in selected_models:
                    models["Random Forest"] = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42)
                if "SVM" in selected_models:
                    models["SVM"] = SVC(probability=True, random_state=42)
                
                # Train all models and store results
                model_results = {}
                for name, model in models.items():
                    model.fit(X_class, y_class)
                    predictions = model.predict(X_class)
                    accuracy = accuracy_score(y_class, predictions)
                    cv_scores = cross_val_score(model, X_class, y_class, cv=3)
                    
                    model_results[name] = {
                        "model": model,
                        "accuracy": accuracy,
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std(),
                        "predictions": predictions
                    }
            
                # Display results
                st.subheader("🎯 Model Comparison Results")
                
                # Model performance comparison
                st.subheader("📊 Model Performance Comparison")
                comparison_data = []
                for name, results in model_results.items():
                    comparison_data.append({
                        "Model": name,
                        "Training Accuracy": f"{results['accuracy']:.2%}",
                        "CV Mean": f"{results['cv_mean']:.2%}",
                        "CV Std": f"{results['cv_std']:.3f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best model selection
                best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['cv_mean'])
                best_model = model_results[best_model_name]['model']
                st.success(f"🏆 Best performing model: **{best_model_name}** (CV Score: {model_results[best_model_name]['cv_mean']:.2%})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Risk Distribution:**")
                    risk_counts = df_encoded["HighRisk"].value_counts()
                    risk_labels = ["Low Risk", "High Risk"]
                    
                    fig_risk = px.pie(
                        values=risk_counts.values,
                        names=risk_labels,
                        title="Risk Distribution in Dataset"
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with col2:
                    st.write("**Dataset Information:**")
                    st.write(f"**Risk Threshold:** {threshold_value:.2f} per 100k")
                    st.write(f"**High Risk Records:** {risk_counts[1]} ({risk_counts[1]/len(df_encoded):.1%})")
                    st.write(f"**Low Risk Records:** {risk_counts[0]} ({risk_counts[0]/len(df_encoded):.1%})")
                    st.write(f"**Models Trained:** {len(model_results)}")
            
                # Feature importance comparison (for tree-based models)
                st.subheader("📊 Feature Importance Comparison")
                tree_models = {name: results for name, results in model_results.items() 
                             if hasattr(results['model'], 'feature_importances_')}
                
                if tree_models:
                    importance_data = []
                    for name, results in tree_models.items():
                        for i, feature in enumerate(['Age', 'Sex', 'Race']):
                            importance_data.append({
                                'Model': name,
                                'Feature': feature,
                                'Importance': results['model'].feature_importances_[i]
                            })
                    
                    importance_df = pd.DataFrame(importance_data)
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        color='Model',
                        orientation='h',
                        title="Feature Importance Comparison (Tree-based Models)",
                        barmode='group'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance is only available for tree-based models (Decision Tree, Random Forest).")
            
                # Decision tree visualization (if Decision Tree is selected)
                if "Decision Tree" in model_results:
                    st.subheader("🌳 Decision Tree Visualization")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(
                        model_results["Decision Tree"]['model'], 
                        feature_names=['Age', 'Sex', 'Race'],
                        class_names=['Low Risk', 'High Risk'],
                        filled=True,
                        ax=ax
                    )
                    plt.title("Decision Tree for Suicide Risk Prediction")
                    st.pyplot(fig)
            
                # Prediction examples using best model
                st.subheader(f"🔮 Risk Predictions by Demographics (using {best_model_name})")
                prediction_results = []
                
                for age_val in df_encoded["Age_encoded"].unique():
                    for sex_val in df_encoded["Sex_encoded"].unique():
                        for race_val in df_encoded["Race_encoded"].unique():
                            pred = best_model.predict([[age_val, sex_val, race_val]])[0]
                            if hasattr(best_model, 'predict_proba'):
                                prob = best_model.predict_proba([[age_val, sex_val, race_val]])[0]
                                prob_text = f"{prob[1]:.2%}"
                            else:
                                prob_text = "N/A"
                            
                            age_label = le_age.inverse_transform([age_val])[0]
                            sex_label = le_sex.inverse_transform([sex_val])[0]
                            race_label = le_race.inverse_transform([race_val])[0]
                            
                            prediction_results.append({
                                "Age Group": age_label,
                                "Sex": sex_label,
                                "Race": race_label,
                                "Predicted Risk": "High Risk" if pred == 1 else "Low Risk",
                                "High Risk Probability": prob_text
                            })
                
                pred_df = pd.DataFrame(prediction_results)
                st.dataframe(pred_df, use_container_width=True)
            
                # Export classification results
                st.subheader("📤 Export Classification Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Model comparison export
                    st.markdown(create_download_link(comparison_df, "model_comparison", "csv"), unsafe_allow_html=True)
                
                with col2:
                    # Predictions export
                    st.markdown(create_download_link(pred_df, "risk_predictions", "csv"), unsafe_allow_html=True)
                
                with col3:
                    # Feature importance export (if available)
                    if tree_models:
                        st.markdown(create_download_link(importance_df, "feature_importance_comparison", "csv"), unsafe_allow_html=True)
                    else:
                        st.write("No tree-based models selected")

# Insights & Summary Section
elif analysis_option == "Insights & Summary":
    st.header("💡 Insights & Summary")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the 'Data Upload & Overview' section.")
    else:
        df = st.session_state.df
        
        st.subheader("📈 Key Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            highest_rate = df.loc[df["Rate_per_100k"].idxmax()]
            st.metric(
                "Highest Rate Group",
                f"{highest_rate['Rate_per_100k']:.1f}/100k",
                f"{highest_rate['AgeGroup']}, {highest_rate['Sex']}, {highest_rate['Race']}"
            )
        
        with col2:
            lowest_rate = df.loc[df["Rate_per_100k"].idxmin()]
            st.metric(
                "Lowest Rate Group", 
                f"{lowest_rate['Rate_per_100k']:.1f}/100k",
                f"{lowest_rate['AgeGroup']}, {lowest_rate['Sex']}, {lowest_rate['Race']}"
            )
        
        with col3:
            total_suicides = df["Suicides"].sum()
            st.metric("Total Suicides", f"{total_suicides:,}")
        
        with col4:
            total_population = df["Population"].sum()
            overall_rate = (total_suicides / total_population) * 100000
            st.metric("Overall Rate", f"{overall_rate:.1f}/100k")
        
        st.subheader("🎯 Key Insights")
        
        # Age group insights
        age_analysis = df.groupby("AgeGroup")["Rate_per_100k"].mean().sort_values(ascending=False)
        st.write("**Age Group Analysis:**")
        st.write(f"• Highest risk age group: **{age_analysis.index[0]}** ({age_analysis.iloc[0]:.1f} per 100k)")
        st.write(f"• Lowest risk age group: **{age_analysis.index[-1]}** ({age_analysis.iloc[-1]:.1f} per 100k)")
        
        # Sex analysis
        sex_analysis = df.groupby("Sex")["Rate_per_100k"].mean()
        sex_ratio = sex_analysis.max() / sex_analysis.min()
        st.write("**Sex Analysis:**")
        st.write(f"• Male vs Female rate ratio: **{sex_ratio:.1f}:1**")
        st.write(f"• Higher risk sex: **{sex_analysis.idxmax()}** ({sex_analysis.max():.1f} per 100k)")
        
        # Race analysis
        race_analysis = df.groupby("Race")["Rate_per_100k"].mean().sort_values(ascending=False)
        st.write("**Race Analysis:**")
        st.write(f"• Highest risk race: **{race_analysis.index[0]}** ({race_analysis.iloc[0]:.1f} per 100k)")
        st.write(f"• Lowest risk race: **{race_analysis.index[-1]}** ({race_analysis.iloc[-1]:.1f} per 100k)")
        
        st.subheader("📊 Data Quality Assessment")
        
        # Data completeness
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        st.write("**Data Completeness:**")
        for col, comp in completeness.items():
            if comp < 100:
                st.write(f"• {col}: {comp:.1f}% complete")
            else:
                st.write(f"• {col}: ✅ Complete")
        
        # Data distribution
        st.write("**Data Distribution:**")
        st.write(f"• Age groups covered: {df['AgeGroup'].nunique()}")
        st.write(f"• Racial groups covered: {df['Race'].nunique()}")
        st.write(f"• Sex categories: {df['Sex'].nunique()}")
        
        st.subheader("⚠️ Important Considerations")
        st.warning("""
        **Data Interpretation Guidelines:**
        - These statistics reflect reported data and may not capture all cases
        - Suicide rates can vary significantly by geographic location and time period
        - Multiple factors beyond demographics influence suicide risk
        - This analysis is for statistical understanding only and should not be used for individual risk assessment
        """)
        
        st.info("""
        **Recommended Actions:**
        - Focus prevention efforts on highest-risk demographic groups
        - Consider additional data sources for comprehensive analysis
        - Implement targeted interventions based on identified patterns
        - Regular monitoring and updating of statistics
        """)
        
        # Export insights summary
        st.subheader("📤 Export Summary Report")
        
        # Create comprehensive summary report
        summary_data = {
            "metric": [
                "Total Records", "Age Groups", "Races", "Average Rate",
                "Highest Rate Group", "Lowest Rate Group", 
                "Total Suicides", "Overall Rate",
                "Highest Risk Age", "Lowest Risk Age",
                "Male vs Female Ratio", "Higher Risk Sex",
                "Highest Risk Race", "Lowest Risk Race"
            ],
            "value": [
                len(df), df["AgeGroup"].nunique(), df["Race"].nunique(), 
                f"{df['Rate_per_100k'].mean():.1f}",
                f"{highest_rate['AgeGroup']}, {highest_rate['Sex']}, {highest_rate['Race']}",
                f"{lowest_rate['AgeGroup']}, {lowest_rate['Sex']}, {lowest_rate['Race']}",
                total_suicides, f"{overall_rate:.1f}",
                age_analysis.index[0], age_analysis.index[-1],
                f"{sex_ratio:.1f}:1", sex_analysis.idxmax(),
                race_analysis.index[0], race_analysis.index[-1]
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.markdown(create_download_link(summary_df, "insights_summary_report", "csv"), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This dashboard is designed for statistical analysis and research purposes. For crisis intervention resources, please contact local emergency services or mental health professionals.*")
