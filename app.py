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
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import numpy as np
import io
import base64
import json
from datetime import datetime
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

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
    ["Data Upload & Overview", "Exploratory Analysis", "Clustering Analysis", "Classification Model", "Statistical Tests", "Predictive Forecasting", "Insights & Summary"]
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

def transform_cdc_format(df):
    """Transform CDC WONDER format to standard format"""
    # Map CDC column names to standard names
    column_mapping = {
        "Year": "Year",
        "Age Group": "AgeGroup", 
        "Gender": "Sex",
        "Race": "Race",
        "Deaths": "Suicides",
        "Population": "Population"
    }
    
    df_transformed = df.rename(columns=column_mapping)
    
    # Clean up gender values
    df_transformed["Sex"] = df_transformed["Sex"].replace({
        "Male": "Male",
        "Female": "Female",
        "M": "Male",
        "F": "Female"
    })
    
    # Clean up age group formatting
    df_transformed["AgeGroup"] = df_transformed["AgeGroup"].str.replace(" years", "").str.replace(" year", "")
    
    return df_transformed

def transform_census_format(df):
    """Transform Census Bureau format to standard format"""
    # Map Census column names to standard names
    column_mapping = {
        "YEAR": "Year",
        "AGEGROUP": "AgeGroup",
        "SEX": "Sex", 
        "RACE": "Race",
        "DEATHS": "Suicides",
        "POPULATION": "Population"
    }
    
    df_transformed = df.rename(columns=column_mapping)
    
    # Clean up sex values (Census typically uses numeric codes)
    sex_mapping = {
        1: "Male", 2: "Female",
        "1": "Male", "2": "Female",
        "MALE": "Male", "FEMALE": "Female"
    }
    df_transformed["Sex"] = df_transformed["Sex"].replace(sex_mapping)
    
    # Clean up race values
    race_mapping = {
        1: "White", 2: "Black", 3: "Hispanic", 4: "Asian", 5: "Other",
        "1": "White", "2": "Black", "3": "Hispanic", "4": "Asian", "5": "Other"
    }
    df_transformed["Race"] = df_transformed["Race"].replace(race_mapping)
    
    return df_transformed

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
        
        # Add CDC/Census data format option
        data_format = st.selectbox(
            "Select Data Format",
            ["Standard Format", "CDC WONDER Format", "Census Bureau Format"],
            help="Choose the format that matches your data source"
        )
        
        if data_format == "Standard Format":
            upload_help = "Your CSV should contain columns: AgeGroup, Sex, Race, Suicides, Population"
            required_cols = ["AgeGroup", "Sex", "Race", "Suicides", "Population"]
        elif data_format == "CDC WONDER Format":
            upload_help = "CDC WONDER format: Year, Age Group, Gender, Race, Deaths, Population"
            required_cols = ["Year", "Age Group", "Gender", "Race", "Deaths", "Population"]
        else:  # Census Bureau Format
            upload_help = "Census format: YEAR, AGEGROUP, SEX, RACE, DEATHS, POPULATION"
            required_cols = ["YEAR", "AGEGROUP", "SEX", "RACE", "DEATHS", "POPULATION"]
        
        uploaded_file = st.file_uploader(
            f"Choose a CSV file with format: {data_format}",
            type="csv",
            help=upload_help
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
            
            # Validate required columns based on format
            missing_columns = [col for col in required_cols if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns for {data_format}: {missing_columns}")
                st.info(upload_help)
            else:
                # Transform data based on selected format
                if data_format == "CDC WONDER Format":
                    df = transform_cdc_format(df)
                elif data_format == "Census Bureau Format":
                    df = transform_census_format(df)
                # Standard format doesn't need transformation
                
                # Ensure we have the standard columns after transformation
                standard_columns = ["AgeGroup", "Sex", "Race", "Suicides", "Population"]
                missing_after_transform = [col for col in standard_columns if col not in df.columns]
                
                if missing_after_transform:
                    st.error(f"Data transformation failed. Missing columns: {missing_after_transform}")
                else:
                    # Clean and validate data
                    df = df.dropna(subset=["Suicides", "Population"])  # Remove rows with missing critical data
                    df = df[df["Population"] > 0]  # Remove rows with zero population to avoid divide by zero
                    
                    st.session_state.df = calculate_rates(df)
                    st.session_state.data_loaded = True
                    st.success(f"Data uploaded and processed successfully! Format: {data_format}")
                    
                    # Show data preview with format info
                    if "Year" in df.columns:
                        year_range = f"{df['Year'].min()}-{df['Year'].max()}"
                        st.info(f"📅 Data covers years: {year_range}")
                    
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please check that your file is a valid CSV and matches the selected format.")
    
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

# Statistical Tests Section
elif analysis_option == "Statistical Tests":
    st.header("📊 Advanced Statistical Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the 'Data Upload & Overview' section.")
    else:
        df = st.session_state.df
        
        st.subheader("🔗 Correlation Analysis")
        
        # Numeric correlation matrix
        numeric_cols = ['Suicides', 'Population', 'Rate_per_100k']
        correlation_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title="Correlation Matrix of Numeric Variables",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig_corr.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Display correlation values
        st.write("**Correlation Coefficients:**")
        st.dataframe(correlation_matrix, use_container_width=True)
        
        st.subheader("🧪 Chi-Square Tests of Independence")
        
        # Chi-square test for categorical variables
        categorical_pairs = [
            ("AgeGroup", "Sex"),
            ("AgeGroup", "Race"), 
            ("Sex", "Race")
        ]
        
        chi_square_results = []
        for var1, var2 in categorical_pairs:
            contingency_table = pd.crosstab(df[var1], df[var2])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            chi_square_results.append({
                "Variable Pair": f"{var1} vs {var2}",
                "Chi-Square Statistic": f"{chi2:.4f}",
                "P-value": f"{p_value:.4f}",
                "Degrees of Freedom": dof,
                "Significant": "Yes" if p_value < 0.05 else "No"
            })
            
            # Display contingency table
            with st.expander(f"Contingency Table: {var1} vs {var2}"):
                st.dataframe(contingency_table, use_container_width=True)
        
        chi_df = pd.DataFrame(chi_square_results)
        st.dataframe(chi_df, use_container_width=True)
        
        st.subheader("📈 ANOVA (Analysis of Variance)")
        
        # ANOVA tests for rate differences across categorical groups
        anova_results = []
        
        for category in ["AgeGroup", "Sex", "Race"]:
            groups = []
            group_names = []
            for group_value in df[category].unique():
                group_data = df[df[category] == group_value]["Rate_per_100k"]
                groups.append(group_data)
                group_names.append(group_value)
            
            # Perform ANOVA
            f_stat, p_value = f_oneway(*groups)
            
            anova_results.append({
                "Category": category,
                "F-Statistic": f"{f_stat:.4f}",
                "P-value": f"{p_value:.4f}",
                "Significant": "Yes" if p_value < 0.05 else "No",
                "Groups": len(groups)
            })
        
        anova_df = pd.DataFrame(anova_results)
        st.dataframe(anova_df, use_container_width=True)
        
        st.subheader("📊 Descriptive Statistics by Groups")
        
        # Detailed group statistics
        for category in ["AgeGroup", "Sex", "Race"]:
            with st.expander(f"Statistics by {category}"):
                group_stats = df.groupby(category)["Rate_per_100k"].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(2)
                group_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']
                st.dataframe(group_stats, use_container_width=True)
        
        st.subheader("📋 Statistical Summary")
        
        # Create summary of findings
        significant_chi_square = chi_df[chi_df["Significant"] == "Yes"]["Variable Pair"].tolist()
        significant_anova = anova_df[anova_df["Significant"] == "Yes"]["Category"].tolist()
        
        st.write("**Key Findings:**")
        
        if significant_chi_square:
            st.write(f"• **Significant associations found:** {', '.join(significant_chi_square)}")
        else:
            st.write("• No significant associations found between categorical variables")
        
        if significant_anova:
            st.write(f"• **Significant rate differences by:** {', '.join(significant_anova)}")
        else:
            st.write("• No significant rate differences found across categorical groups")
        
        # Strongest correlations
        corr_vals = correlation_matrix.abs().unstack().sort_values(ascending=False)
        corr_vals = corr_vals[corr_vals < 1.0]  # Remove self-correlations
        if len(corr_vals) > 0:
            strongest_corr = corr_vals.index[0]
            strongest_val = corr_vals.iloc[0]
            st.write(f"• **Strongest correlation:** {strongest_corr[0]} vs {strongest_corr[1]} (r = {strongest_val:.3f})")
        
        # Export statistical results
        st.subheader("📤 Export Statistical Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Correlation matrix export
            st.markdown(create_download_link(correlation_matrix, "correlation_matrix", "csv"), unsafe_allow_html=True)
        
        with col2:
            # Chi-square results export
            st.markdown(create_download_link(chi_df, "chi_square_tests", "csv"), unsafe_allow_html=True)
        
        with col3:
            # ANOVA results export
            st.markdown(create_download_link(anova_df, "anova_results", "csv"), unsafe_allow_html=True)

# Predictive Forecasting Section
elif analysis_option == "Predictive Forecasting":
    st.header("🔮 Predictive Forecasting & Future Trends")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the 'Data Upload & Overview' section.")
    else:
        df = st.session_state.df
        
        # Check if data has temporal information
        has_year = 'Year' in df.columns
        
        if not has_year:
            st.info("📈 **Trend Analysis Based on Current Data**")
            st.write("Since your data doesn't contain temporal information (Year column), we'll perform trend analysis based on demographic patterns.")
            
            # Demographic trend analysis
            st.subheader("📊 Demographic Risk Predictions")
            
            # Predict risk levels based on current patterns
            df_analysis = df.copy()
            
            # Calculate risk categories based on suicide rates
            rate_percentiles = df_analysis['Rate_per_100k'].quantile([0.33, 0.67])
            def categorize_risk(rate):
                if rate <= rate_percentiles.iloc[0]:
                    return "Low Risk"
                elif rate <= rate_percentiles.iloc[1]:
                    return "Medium Risk"
                else:
                    return "High Risk"
            
            df_analysis['Risk_Category'] = df_analysis['Rate_per_100k'].apply(categorize_risk)
            
            # Risk distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk by demographic
                fig = px.bar(
                    df_analysis.groupby(['AgeGroup', 'Risk_Category']).size().reset_index(name='Count'),
                    x='AgeGroup', y='Count', color='Risk_Category',
                    title='Risk Categories by Age Group',
                    color_discrete_map={
                        'Low Risk': 'green',
                        'Medium Risk': 'orange', 
                        'High Risk': 'red'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk heatmap
                risk_pivot = df_analysis.pivot_table(
                    values='Rate_per_100k', 
                    index='AgeGroup', 
                    columns='Sex', 
                    aggfunc='mean'
                )
                fig = px.imshow(
                    risk_pivot,
                    title='Suicide Rate Heatmap by Age and Sex',
                    color_continuous_scale='Reds',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Predictive modeling using existing demographic patterns
            st.subheader("🎯 Risk Prediction Model")
            
            # Simple trend projection based on current patterns
            age_trends = df_analysis.groupby('AgeGroup')['Rate_per_100k'].mean().sort_values(ascending=False)
            sex_trends = df_analysis.groupby('Sex')['Rate_per_100k'].mean()
            race_trends = df_analysis.groupby('Race')['Rate_per_100k'].mean().sort_values(ascending=False)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Highest Risk Age Groups:**")
                for i, (age, rate) in enumerate(age_trends.head(3).items()):
                    st.write(f"{i+1}. {age}: {rate:.1f} per 100k")
            
            with col2:
                st.write("**Risk by Sex:**")
                for sex, rate in sex_trends.items():
                    st.write(f"• {sex}: {rate:.1f} per 100k")
            
            with col3:
                st.write("**Highest Risk Race Groups:**")
                for i, (race, rate) in enumerate(race_trends.head(3).items()):
                    st.write(f"{i+1}. {race}: {rate:.1f} per 100k")
            
            # Future scenario projection
            st.subheader("📈 Scenario Projections")
            
            scenario = st.selectbox(
                "Select projection scenario:",
                ["Current Trends Continue", "Optimistic (20% reduction)", "Pessimistic (15% increase)"]
            )
            
            multiplier = {"Current Trends Continue": 1.0, "Optimistic (20% reduction)": 0.8, "Pessimistic (15% increase)": 1.15}[scenario]
            
            df_projected = df_analysis.copy()
            df_projected['Projected_Rate'] = df_projected['Rate_per_100k'] * multiplier
            df_projected['Projected_Suicides'] = (df_projected['Projected_Rate'] / 100000) * df_projected['Population']
            
            # Comparison chart
            comparison_data = pd.DataFrame({
                'Demographic': df_projected['AgeGroup'] + ' - ' + df_projected['Sex'],
                'Current Rate': df_projected['Rate_per_100k'],
                'Projected Rate': df_projected['Projected_Rate']
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current Rate',
                x=comparison_data['Demographic'],
                y=comparison_data['Current Rate'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Projected Rate',
                x=comparison_data['Demographic'], 
                y=comparison_data['Projected Rate'],
                marker_color='darkblue'
            ))
            fig.update_layout(
                title=f'Current vs Projected Rates - {scenario}',
                xaxis_title='Demographic Group',
                yaxis_title='Rate per 100k',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Time series analysis for data with years
            st.info("📈 **Advanced Time Series Forecasting**")
            st.write("Your data contains temporal information. Performing statistical time series forecasting with confidence intervals.")
            
            # Data preprocessing and validation
            df_ts = df.copy()
            df_ts['Year'] = pd.to_numeric(df_ts['Year'], errors='coerce')
            df_ts = df_ts.dropna(subset=['Year'])
            df_ts = df_ts.sort_values('Year')
            
            # Forecasting configuration
            st.subheader("🔧 Forecast Configuration")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                forecast_horizon = st.selectbox(
                    "Forecast Horizon (years):",
                    [3, 5, 10, 15],
                    index=1
                )
            
            with col2:
                demographic_grouping = st.selectbox(
                    "Forecast by demographic:",
                    ["Overall", "By Sex", "By Age Group", "By Race"]
                )
            
            with col3:
                model_selection = st.selectbox(
                    "Model Selection:",
                    ["Automatic (Best AIC)", "ARIMA", "Exponential Smoothing"]
                )
            
            # Prepare time series data based on grouping
            if demographic_grouping == "Overall":
                ts_data = df_ts.groupby('Year').agg({
                    'Suicides': 'sum',
                    'Population': 'sum'
                }).reset_index()
                ts_data['Rate_per_100k'] = (ts_data['Suicides'] / ts_data['Population']) * 100000
                forecast_groups = [("Overall", ts_data)]
            else:
                group_col = {"By Sex": "Sex", "By Age Group": "AgeGroup", "By Race": "Race"}[demographic_grouping]
                forecast_groups = []
                for group_val in df_ts[group_col].unique():
                    group_data = df_ts[df_ts[group_col] == group_val].groupby('Year').agg({
                        'Suicides': 'sum',
                        'Population': 'sum'
                    }).reset_index()
                    if len(group_data) >= 3:  # Minimum data points for modeling
                        group_data['Rate_per_100k'] = (group_data['Suicides'] / group_data['Population']) * 100000
                        forecast_groups.append((group_val, group_data))
            
            # Helper function for model fitting and forecasting
            def fit_forecast_model(data, horizon, model_type="auto"):
                """Fit time series model and generate forecasts with confidence intervals"""
                if len(data) < 3:
                    return None, None, None, None
                
                # Prepare time series
                ts = data.set_index('Year')['Rate_per_100k']
                
                best_model = None
                best_aic = float('inf')
                models_to_try = []
                
                if model_type == "auto" or model_type == "Automatic (Best AIC)":
                    models_to_try = ["ARIMA", "ETS"]
                elif model_type == "ARIMA":
                    models_to_try = ["ARIMA"]
                else:  # Exponential Smoothing
                    models_to_try = ["ETS"]
                
                # Try different models
                for model_name in models_to_try:
                    try:
                        if model_name == "ARIMA":
                            # Try simple ARIMA models
                            for p, d, q in [(1,1,1), (2,1,1), (1,1,2), (0,1,1)]:
                                try:
                                    model = ARIMA(ts, order=(p,d,q))
                                    fitted = model.fit()
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_model = fitted
                                        best_model_name = f"ARIMA({p},{d},{q})"
                                except:
                                    continue
                        
                        elif model_name == "ETS":
                            try:
                                model = ETSModel(ts, trend="add", seasonal=None)
                                fitted = model.fit()
                                if fitted.aic < best_aic:
                                    best_aic = fitted.aic
                                    best_model = fitted
                                    best_model_name = "ETS"
                            except:
                                continue
                    except:
                        continue
                
                # Generate forecast if we have a model
                if best_model is not None:
                    try:
                        forecast = best_model.forecast(steps=horizon)
                        # Get confidence intervals
                        pred_summary = best_model.get_prediction(start=0, end=len(ts) + horizon - 1)
                        conf_int = pred_summary.conf_int()
                        
                        return best_model, forecast, conf_int, best_model_name
                    except:
                        return None, None, None, None
                
                return None, None, None, None
            
            # Perform forecasting for each group
            forecast_results = {}
            forecast_charts = []
            
            for group_name, group_data in forecast_groups:
                if len(group_data) >= 3:  # Minimum data for forecasting
                    model, forecast, conf_int, model_name = fit_forecast_model(
                        group_data, forecast_horizon, model_selection
                    )
                    
                    if model is not None and forecast is not None:
                        # Prepare forecast data
                        last_year = int(group_data['Year'].max())
                        future_years = list(range(last_year + 1, last_year + forecast_horizon + 1))
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'Year': future_years,
                            'Forecast': forecast.values if hasattr(forecast, 'values') else forecast,
                            'Group': group_name,
                            'Model': model_name
                        })
                        
                        # Add confidence intervals if available
                        if conf_int is not None:
                            try:
                                forecast_lower = conf_int.iloc[-forecast_horizon:, 0].values
                                forecast_upper = conf_int.iloc[-forecast_horizon:, 1].values
                                forecast_df['Lower_CI'] = forecast_lower
                                forecast_df['Upper_CI'] = forecast_upper
                            except:
                                pass
                        
                        forecast_results[group_name] = {
                            'data': group_data,
                            'forecast': forecast_df,
                            'model': model_name,
                            'aic': getattr(model, 'aic', None)
                        }
            
            # Display forecasting results
            st.subheader("📈 Forecast Results")
            
            if not forecast_results:
                st.warning("Insufficient data for time series forecasting. Need at least 3 years of data.")
            else:
                # Create tabs for different groups
                if len(forecast_results) == 1:
                    # Single forecast
                    group_name, result = list(forecast_results.items())[0]
                    
                    # Historical data visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=result['data']['Year'], 
                        y=result['data']['Rate_per_100k'],
                        mode='lines+markers', 
                        name='Historical Data',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast line
                    fig.add_trace(go.Scatter(
                        x=result['forecast']['Year'],
                        y=result['forecast']['Forecast'],
                        mode='lines+markers',
                        name=f'Forecast ({result["model"]})',
                        line=dict(color='red', width=2, dash='dot')
                    ))
                    
                    # Confidence intervals if available
                    if 'Lower_CI' in result['forecast'].columns:
                        fig.add_trace(go.Scatter(
                            x=result['forecast']['Year'],
                            y=result['forecast']['Upper_CI'],
                            mode='lines',
                            name='Upper 95% CI',
                            line=dict(color='rgba(255,0,0,0)'),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=result['forecast']['Year'],
                            y=result['forecast']['Lower_CI'],
                            mode='lines',
                            name='95% Confidence Interval',
                            line=dict(color='rgba(255,0,0,0)'),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
                    
                    fig.update_layout(
                        title=f'Time Series Forecast - {group_name}',
                        xaxis_title='Year',
                        yaxis_title='Rate per 100k',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_rate = result['data']['Rate_per_100k'].iloc[-1]
                        st.metric("Current Rate", f"{current_rate:.2f} per 100k")
                    with col2:
                        future_rate = result['forecast']['Forecast'].iloc[-1]
                        change = future_rate - current_rate
                        st.metric(
                            f"Forecast ({forecast_horizon}Y)", 
                            f"{future_rate:.2f} per 100k",
                            f"{change:+.2f}"
                        )
                    with col3:
                        if result['aic']:
                            st.metric("Model AIC", f"{result['aic']:.1f}")
                        st.write(f"**Model:** {result['model']}")
                
                else:
                    # Multiple forecasts - use tabs
                    tabs = st.tabs(list(forecast_results.keys()))
                    for i, (group_name, result) in enumerate(forecast_results.items()):
                        with tabs[i]:
                            # Similar visualization for each group
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=result['data']['Year'], 
                                y=result['data']['Rate_per_100k'],
                                mode='lines+markers', 
                                name='Historical Data',
                                line=dict(color='blue')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=result['forecast']['Year'],
                                y=result['forecast']['Forecast'],
                                mode='lines+markers',
                                name=f'Forecast ({result["model"]})',
                                line=dict(color='red', dash='dot')
                            ))
                            
                            if 'Lower_CI' in result['forecast'].columns:
                                fig.add_trace(go.Scatter(
                                    x=result['forecast']['Year'],
                                    y=result['forecast']['Upper_CI'],
                                    mode='lines',
                                    line=dict(color='rgba(255,0,0,0)'),
                                    showlegend=False
                                ))
                                fig.add_trace(go.Scatter(
                                    x=result['forecast']['Year'],
                                    y=result['forecast']['Lower_CI'],
                                    mode='lines',
                                    fill='tonexty',
                                    fillcolor='rgba(255,0,0,0.2)',
                                    name='95% CI'
                                ))
                            
                            fig.update_layout(
                                title=f'Forecast - {group_name}',
                                xaxis_title='Year',
                                yaxis_title='Rate per 100k'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                current = result['data']['Rate_per_100k'].iloc[-1]
                                future = result['forecast']['Forecast'].iloc[-1]
                                st.metric("Current", f"{current:.2f}")
                                st.metric("Forecast", f"{future:.2f}")
                            with col2:
                                if result['aic']:
                                    st.write(f"**AIC:** {result['aic']:.1f}")
                                st.write(f"**Model:** {result['model']}")
                
                # Model comparison summary
                if len(forecast_results) > 1:
                    st.subheader("📊 Model Comparison Summary")
                    comparison_data = []
                    for group_name, result in forecast_results.items():
                        current_rate = result['data']['Rate_per_100k'].iloc[-1]
                        forecast_rate = result['forecast']['Forecast'].iloc[-1]
                        change = forecast_rate - current_rate
                        comparison_data.append({
                            'Group': group_name,
                            'Current Rate': current_rate,
                            'Forecast Rate': forecast_rate,
                            'Change': change,
                            'Model': result['model'],
                            'AIC': result.get('aic', 'N/A')
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
        
        # Export forecasting results
        st.subheader("📤 Export Forecasting Results")
        
        if has_year and forecast_results:
            # Export enhanced time series forecasts with confidence intervals
            all_forecasts = []
            for group_name, result in forecast_results.items():
                # Historical data
                hist_data = result['data'].copy()
                hist_data['Group'] = group_name
                hist_data['Type'] = 'Historical'
                hist_data['Model'] = result['model']
                hist_data = hist_data.rename(columns={'Rate_per_100k': 'Value'})
                
                # Forecast data
                forecast_data = result['forecast'].copy()
                forecast_data['Type'] = 'Forecast'
                forecast_data = forecast_data.rename(columns={'Forecast': 'Value'})
                
                # Select common columns
                hist_cols = ['Year', 'Group', 'Type', 'Value', 'Model']
                forecast_cols = ['Year', 'Group', 'Type', 'Value', 'Model']
                if 'Lower_CI' in forecast_data.columns:
                    forecast_cols.extend(['Lower_CI', 'Upper_CI'])
                
                all_forecasts.append(hist_data[hist_cols])
                all_forecasts.append(forecast_data[forecast_cols])
            
            combined_forecast_df = pd.concat(all_forecasts, ignore_index=True)
            st.markdown(create_download_link(combined_forecast_df, "advanced_time_series_forecast", "csv"), unsafe_allow_html=True)
        else:
            # Export demographic projections
            projection_df = df_projected[['AgeGroup', 'Sex', 'Race', 'Rate_per_100k', 'Projected_Rate', 'Projected_Suicides']].copy()
            st.markdown(create_download_link(projection_df, "demographic_projections", "csv"), unsafe_allow_html=True)

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
