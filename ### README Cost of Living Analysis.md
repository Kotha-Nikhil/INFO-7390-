### **README: Cost of Living Analysis**  

#### **Project Title**  
**Decoding the Cost of Living: Data Insights and Predictive Modeling**  

#### **Project Description**  
This project analyzes **cost of living** data using **data preprocessing, missing data imputation, feature selection, and predictive modeling** techniques. The goal is to identify key factors influencing living costs, evaluate missing data handling methods, and build models to predict cost variations across different regions.  

#### **Key Features**  
- **Exploratory Data Analysis (EDA)**: Understanding cost distributions and relationships.  
- **Missing Data Imputation**: Comparing **mean, median, and mode imputation** at 1%, 5%, and 10% missing values.  
- **Feature Selection**: Using **correlation analysis, feature importance, and recursive feature elimination (RFE)**.  
- **Predictive Modeling**: Training **regression and classification models** to analyze living cost trends.  
- **Model Evaluation**: Measuring accuracy, bias, and variance across different imputation techniques.  

#### **Installation & Dependencies**  
Ensure you have the following Python libraries installed:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

#### **Dataset Source**  
The dataset contains key cost metrics such as **housing, food, transportation, healthcare, and taxes**. It is sourced from public economic databases and cost of living indices.  

#### **How to Use This Project**  
1. **Load the dataset** and perform **EDA** to understand cost trends.  
2. **Handle missing data** using different imputation strategies.  
3. **Apply feature selection** to identify the most relevant predictors.  
4. **Train machine learning models** for regression and classification.  
5. **Evaluate models** and compare imputation techniques.  

#### **Results & Insights**  
- **Median imputation** was the most stable method for handling missing data.  
- **Housing, food, and childcare costs** were the most influential factors in predicting total cost.  
- **Logistic Regression achieved 67.7% accuracy** in classifying metro vs. non-metro areas, but additional features could improve performance.  
- Removing **outliers** helped improve model generalization and reduced bias.  

#### **References**  
- Bureau of Labor Statistics (2023). *Consumer Expenditure Surveys*. [Link](https://www.bls.gov/cex/)  
- Little, R.J.A., & Rubin, D.B. (2019). *Statistical Analysis with Missing Data (3rd ed.)*. Wiley.  
- Breiman, L. (2001). *Random Forests*. *Machine Learning, 45*(1), 5-32.  
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research, 12*, 2825–2830.  

