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
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import io
import base64

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

# Classification Model Section
elif analysis_option == "Classification Model":
    st.header("🤖 Classification Model")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first in the 'Data Upload & Overview' section.")
    else:
        df = st.session_state.df.copy()
        df_encoded, le_age, le_sex, le_race = encode_categorical_features(df)
        
        st.subheader("⚙️ Model Configuration")
        col1, col2 = st.columns(2)
        
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
            max_depth = st.slider("Maximum Tree Depth", min_value=2, max_value=10, value=3)
        
        if st.button("Train Classification Model", type="primary"):
            # Create high/low risk labels
            threshold_value = df_encoded["Rate_per_100k"].quantile(risk_threshold)
            df_encoded["HighRisk"] = (df_encoded["Rate_per_100k"] > threshold_value).astype(int)
            
            # Prepare features and target
            features = ["Age_encoded", "Sex_encoded", "Race_encoded"]
            X_class = df_encoded[features]
            y_class = df_encoded["HighRisk"]
            
            # Train classifier
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            clf.fit(X_class, y_class)
            
            # Display results
            st.subheader("🎯 Model Results")
            
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
                st.write("**Classification Accuracy:**")
                predictions = clf.predict(X_class)
                accuracy = (predictions == y_class).mean()
                st.metric("Training Accuracy", f"{accuracy:.2%}")
                
                st.write(f"**Risk Threshold:** {threshold_value:.2f} per 100k")
                st.write(f"**High Risk Records:** {risk_counts[1]} ({risk_counts[1]/len(df_encoded):.1%})")
                st.write(f"**Low Risk Records:** {risk_counts[0]} ({risk_counts[0]/len(df_encoded):.1%})")
            
            # Feature importance
            st.subheader("📊 Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Sex', 'Race'],
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Predicting Suicide Risk"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Decision tree visualization
            st.subheader("🌳 Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(
                clf, 
                feature_names=['Age', 'Sex', 'Race'],
                class_names=['Low Risk', 'High Risk'],
                filled=True,
                ax=ax
            )
            plt.title("Decision Tree for Suicide Risk Prediction")
            st.pyplot(fig)
            
            # Prediction examples
            st.subheader("🔮 Risk Predictions by Demographics")
            prediction_results = []
            
            for age_val in df_encoded["Age_encoded"].unique():
                for sex_val in df_encoded["Sex_encoded"].unique():
                    for race_val in df_encoded["Race_encoded"].unique():
                        pred = clf.predict([[age_val, sex_val, race_val]])[0]
                        prob = clf.predict_proba([[age_val, sex_val, race_val]])[0]
                        
                        age_label = le_age.inverse_transform([age_val])[0]
                        sex_label = le_sex.inverse_transform([sex_val])[0]
                        race_label = le_race.inverse_transform([race_val])[0]
                        
                        prediction_results.append({
                            "Age Group": age_label,
                            "Sex": sex_label,
                            "Race": race_label,
                            "Predicted Risk": "High Risk" if pred == 1 else "Low Risk",
                            "High Risk Probability": f"{prob[1]:.2%}"
                        })
            
            pred_df = pd.DataFrame(prediction_results)
            st.dataframe(pred_df, use_container_width=True)

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

# Footer
st.markdown("---")
st.markdown("*This dashboard is designed for statistical analysis and research purposes. For crisis intervention resources, please contact local emergency services or mental health professionals.*")
