# COVID
#                Analyzing the Influence of Socioeconomic and Environmental Factors on COVID-19 and Influenza
## OVERVIEW
This project aims to explore the relationship between socioeconomic and environmental factors and the spread and mortality rates of COVID-19 and Influenza. By analyzing data from a range of countries, the study examines how factors such as GDP, healthcare spending, population density, and climate conditions impact disease outcomes across different contexts. To provide a comprehensive analysis, countries are categorized into four groups: DEVELOPED, DEVELOPING, UNDERDEVELOPED, and ISLAND allowing for a nuanced understanding of how these factors affect each group differently.Assess how socioeconomic and environmental factors influence the outcomes of COVID-19 and Influenza across a diverse set of countries.

## Features
- **Data Loading and Preprocessing**: Loads and cleans data to prepare for analysis, handling missing values and data transformations.
- **Exploratory Data Analysis (EDA)**: Conducts EDA to understand data distributions and relationships between variables.
- **Model Implementation**: Implements statistical models to assess the influence of socioeconomic and environmental factors on disease outcomes.
- **Visualization**: Generates visualizations to illustrate the findings from the data analysis and model results.

## Installation
To set up this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>

   python main.py

## Dataset
The project uses publicly available datasets for COVID-19 and Influenza, which include:

COVID-19 Data: Case counts, mortality rates, and other relevant metrics.

Influenza Data: case counts and mortality rates.

Socioeconomic Data and healthcare Data: GDP, population, population density, Latitute etc.

Climate Data: Climate conditions, average temperatures, etc.

Healthcare Data : Healthcare Expenditure

## Necessary Libraries to run this project

- `pandas`: For data manipulation and analysis
- `numpy`: For numerical computations
- `matplotlib`: For data visualization
- `seaborn`: For statistical data visualization
- `xgboost`: For gradient boosting and building predictive models
- `scikit-learn`: For machine learning algorithms and utilities

## DATA PREPROCESSING
Preprocess Data Clean and preprocess the data, which may involve handling missing values, normalizing data, or encoding categorical variables.

## EDA 
 Exploratory Data Analysis (EDA) was used to gain insights into the dataset before modeling. This involved data cleaning to handle missing values and correct anomalies, feature analysis to explore distributions and relationships through visualizations like histograms and correlation matrices, to prepare variables for modeling. The data was segmented into categories such as developed, developing, underdeveloped, and island nations to tailor predictions. EDA helped identify key trends and patterns in COVID-19 and Influenza outcomes, guiding feature selection and informing the overall modeling approach, ultimately enhancing the accuracy of the predictions.
  

# IMPLEMENTING MODEL 
 This project implemented three machine learning models—XGBoost, Bagging Regressor, and Voting Regressor—to predict COVID-19 and Influenza-related variables across different country categories. Data preprocessing involved handling missing values, one-hot encoding categorical variables, and filtering relevant features.
XGBoost is a gradient boosting model known for efficiency and scalability, evaluated using metrics like RMSE, MAE, and R² with cross-validation to assess accuracy.
Bagging Regressor uses Decision Trees to build multiple models on different data subsets, enhancing robustness. Performance was measured using the same metrics, with visualizations showing predicted vs. true values.
Voting Regressor combines predictions from Decision Tree, Random Forest, and XGBoost to improve accuracy. All models underwent hyperparameter tuning via GridSearchCV, and their performance was evaluated across developed, developing, underdeveloped, and island nations, using metrics to compare prediction accuracy in various scenarios.

## Examples

### Example Workflow

Follow the steps below to run a complete analysis from importing libraries to implementing the model:

1. **Import Libraries**
   Ensure all necessary libraries are imported at the beginning of the script. This will include libraries for data manipulation, visualization, and modeling.

   ```python
   # Import necessary libraries
   import pandas as pd
   import numpy as np
   import geopandas as gpd
   import matplotlib.pyplot as plt
   import seaborn as sns
   import xgboost as xgb
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error, r2_score
   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
   from sklearn.ensemble import BaggingClassifier, VotingClassifier
   from xgboost import XGBClassifier
   from sklearn.experimental import enable_iterative_imputer
   from sklearn.impute import IterativeImputer
   from sklearn.model_selection import KFold
   from sklearn.model_selection import cross_val_score
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.ensemble import BaggingRegressor
   from sklearn.ensemble import RandomForestRegressor, VotingRegressor
   from xgboost import XGBRegressor
   from sklearn.model_selection import train_test_split, KFold, cross_val_score
   from sklearn.ensemble import VotingRegressor, RandomForestRegressor
   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.impute import KNNImputer

   
  
2.**Load the dataset from your data source, such as a CSV file and after that preprocess the data and EDA**
   
       # Load the neccessary datasets.
        covid_file_path = '/content/drive/MyDrive/FINAL PROJECT/covid_data.csv'
        influenza_file_path = '/content/drive/My Drive/FINAL PROJECT/influenza_data.csv'
        
        #IMPUTATION BY SAMPLING IS USED TO HANDLE MISSSING VALUES
     #def impute_with_sampling(df, column):
        non_zero_non_na_values = df.loc[(df[column] != 0) & (~df[column].isna()), column]
        df.loc[df[column].isna() | (df[column] == 0), column] = np.random.choice(non_zero_non_na_values, size=df.loc[df[column].isna() | (df[column] == 0), 
         column].shape[0], replace=True)
         return df

         
        #EDA
         # Correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        plt.figure(figsize=(14, 10))
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidths=0.5)
        plt.title(f'Correlation Matrix for {category_name} Countries')
        plt.show()
    else:
        print("No numeric data available for correlation matrix.")



# DATASET USED FOR THE PROJECT #

 COVID - https://www.kaggle.com/datasets/georgesaavedra/covid19-dataset

 INFLUENZA - https://ourworldindata.org/explorers/influenza?facet=none&Confirmed+cases+or+Symptoms=Confirmed+cases&Metric=Share+of+positive+tests+%28%25%29&Interval=Monthly&Surveillance+type=All+types&country=Northern+Hemisphere~Southern+Hemisphere

 HEALTH EXPENDITURE - https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS

 CLIMATE - https://www.kaggle.com/datasets/goyaladi/climate-insights-dataset?select=climate_change_data.csv

 SOCIOECONOMIC - https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023



        

  
      
       
   








