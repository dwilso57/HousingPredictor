# Suicide Data Analysis Dashboard

## Overview

This is a Streamlit-based data analysis dashboard designed to analyze suicide statistics by demographic factors including age, sex, and race. The application provides comprehensive data visualization, machine learning clustering, and predictive modeling capabilities for understanding patterns in suicide data. It combines interactive web interface with statistical analysis tools to help researchers and analysts explore demographic trends and build insights from suicide statistics data.

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
- **Supervised Learning**: Decision Tree classifier for predictive modeling
- **Model Evaluation**: Confusion matrix and classification reports for model performance assessment
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
  - LabelEncoder for categorical data preprocessing
  - KMeans for clustering analysis
  - DecisionTreeClassifier for classification
  - Metrics modules for model evaluation

### Utility Libraries
- **Base64**: Data encoding for file handling
- **IO**: Input/output operations for data processing

### Data Sources (Referenced)
- **CDC Data**: Centers for Disease Control and Prevention suicide statistics
- **Census Data**: US Census Bureau population demographics
- **Note**: Application includes sample data structure but expects real data uploads for production use