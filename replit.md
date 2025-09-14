# Suicide Data Analysis Dashboard

## Overview

This is a Streamlit-based data analysis dashboard designed to analyze suicide statistics by demographic factors including age, sex, and race. The application provides comprehensive data visualization, machine learning clustering, and predictive modeling capabilities for understanding patterns in suicide data. It combines interactive web interface with statistical analysis tools to help researchers and analysts explore demographic trends and build insights from suicide statistics data.

## Recent Changes

### December 2025 - Comprehensive Feature Completion
- **Advanced Predictive Forecasting**: Implemented statistical time series forecasting with ARIMA and ETS models, automatic AIC-based model selection, 95% confidence intervals, configurable forecast horizons (3-15 years), and demographic grouping capabilities
- **Enhanced Machine Learning**: Added Random Forest and SVM classifiers with cross-validation, model comparison metrics, and comprehensive performance evaluation
- **Statistical Analysis Suite**: Integrated correlation matrices, chi-square tests, ANOVA analysis, and advanced statistical visualizations
- **Multi-Format Data Integration**: Added support for CDC WONDER and Census Bureau data formats with automatic transformation and validation
- **Comprehensive Export System**: Implemented CSV download functionality across all analysis sections with full data export capabilities

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Single-page application with sidebar navigation providing different analysis views
- **Interactive Dashboard**: Multi-section interface including data upload, exploratory analysis, clustering, classification modeling, and insights summary
- **Session State Management**: Persistent data storage across user interactions using Streamlit's session state
- **Responsive Layout**: Wide layout configuration with expandable sidebar for optimal data visualization

### Data Processing Pipeline
- **Data Input**: Support for CSV file uploads with fallback to sample demonstration data
- **Data Preprocessing**: Automatic calculation of suicide rates per 100,000 population
- **Feature Engineering**: Label encoding for categorical variables (age groups, sex, race) to enable machine learning operations
- **Data Validation**: Built-in data structure validation and error handling

### Visualization Architecture
- **Multiple Visualization Libraries**: 
  - Plotly Express and Graph Objects for interactive web-based charts
  - Matplotlib and Seaborn for statistical plotting
  - Integrated subplot functionality for complex multi-chart displays
- **Chart Types**: Bar charts, scatter plots, heatmaps, and decision tree visualizations
- **Dynamic Filtering**: Interactive controls for exploring different demographic segments

### Machine Learning Components
- **Unsupervised Learning**: K-Means clustering for demographic pattern discovery
- **Supervised Learning**: Multiple classifiers including Decision Tree, Random Forest, and SVM with cross-validation
- **Model Evaluation**: Comprehensive comparison metrics, confusion matrices, classification reports, and feature importance analysis
- **Time Series Forecasting**: Advanced ARIMA and ETS models with automatic model selection, confidence intervals, and forecast visualization
- **Scikit-learn Integration**: Standardized ML pipeline using sklearn preprocessing and algorithms

### Data Architecture
- **In-Memory Processing**: Pandas DataFrame-based data management
- **Sample Data Generation**: Built-in demonstration dataset mimicking CDC and Census data structure
- **Rate Calculation Engine**: Automatic computation of standardized suicide rates per population

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for creating the interactive dashboard
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing support

### Visualization Libraries
- **Plotly**: Interactive plotting (plotly.express, plotly.graph_objects, plotly.subplots)
- **Matplotlib**: Static plotting capabilities
- **Seaborn**: Statistical data visualization

### Machine Learning Libraries
- **Scikit-learn**: Complete machine learning toolkit including:
  - Multiple classifiers (Decision Tree, Random Forest, SVM)
  - LabelEncoder for categorical data preprocessing
  - KMeans for clustering analysis
  - Cross-validation and model evaluation metrics
- **Statsmodels**: Advanced statistical modeling including:
  - ARIMA time series modeling
  - ETS (Exponential Smoothing) models
  - Statistical tests and model diagnostics
- **SciPy**: Statistical functions and hypothesis testing

### Utility Libraries
- **Base64**: Data encoding for file handling and CSV exports
- **IO**: Input/output operations for data processing
- **Warnings**: Error handling for statistical model fitting

### Data Sources Integration
- **CDC WONDER Format**: Automated transformation of CDC data with Year/Age Group/Gender/Race/Deaths/Population structure
- **Census Bureau Format**: Support for YEAR/AGEGROUP/SEX/RACE/DEATHS/POPULATION with numeric code mapping
- **Standard Format**: Original AgeGroup/Sex/Race/Suicides/Population structure
- **Sample Data**: Built-in demonstration dataset for testing and exploration