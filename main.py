import random
from PIL import Image
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
import joblib
import warnings
warnings.filterwarnings("ignore") 


st.set_option('deprecation.showPyplotGlobalUse', False)
# Set page layout
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .header {
        padding: 10px;
        background-color: #85d0df;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .menu {
        background-color: #98e080;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

train_original = pd.read_csv('train.csv')

test_original = pd.read_csv('test.csv')

full_data = pd.concat([train_original, test_original], axis=0)

full_data = full_data.sample(frac=1).reset_index(drop=True)


def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()

# Function to simulate Credit Risk Assessment
def simulate_credit_risk_assessment():
    st.markdown("<div class='header'>Credit Risk Assessment</div>", unsafe_allow_html=True)

    # User input for credit assessment
    user_name = st.text_input("Enter Applicant Name:")
    

    # Detect fraud
    detect_fraud()

# Function to calculate a random credit score (for simulation purposes)
def calculate_credit_score():
    return random.randint(300, 850)

def value_cnt_norm_cal(df,feature):
    '''
    Function to calculate the count of each value in a feature and normalize it
    '''
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_outliers = ['Family member count','Income', 'Employment length']):
        self.feat_with_outliers = feat_with_outliers
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_outliers).issubset(df.columns)):
            # 25% quantile
            Q1 = df[self.feat_with_outliers].quantile(.25)
            # 75% quantile
            Q3 = df[self.feat_with_outliers].quantile(.75)
            IQR = Q3 - Q1
            # keep the data within 1.5 IQR
            df = df[~((df[self.feat_with_outliers] < (Q1 - 3 * IQR)) |(df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['Has a mobile phone','Children count','Job title','Account age']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days = ['Employment length', 'Age']):
        self.feat_with_days = feat_with_days
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if (set(self.feat_with_days).issubset(X.columns)):
            # convert days to absolute value
            X[['Employment length','Age']] = np.abs(X[['Employment length','Age']])
            return X
        else:
            print("One or more features are not in the dataframe")
            return X


class RetireeHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, df):
        return self
    def transform(self, df):
        if 'Employment length' in df.columns:
            # select rows with employment length is 365243 which corresponds to retirees
            df_ret_idx = df['Employment length'][df['Employment length'] == 365243].index
            # change 365243 to 0
            df.loc[df_ret_idx,'Employment length'] = 0
            return df
        else:
            print("Employment length is not in the dataframe")
            return df


class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_skewness=['Income','Age']):
        self.feat_with_skewness = feat_with_skewness
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_skewness).issubset(df.columns)):
            # Handle skewness with cubic root transformation
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_num_enc=['Has a work phone','Has a phone','Has an email']):
        self.feat_with_num_enc = feat_with_num_enc
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_num_enc).issubset(df.columns)):
            # Change 0 to N and 1 to Y for all the features in feat_with_num_enc
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1:'Y',0:'N'})
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class OneHotWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,one_hot_enc_ft = ['Gender', 'Marital status', 'Dwelling', 'Employment status', 'Has a car', 'Has a property', 'Has a work phone', 'Has a phone', 'Has an email']):
        self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.one_hot_enc_ft).issubset(df.columns)):
            # function to one hot encode the features in one_hot_enc_ft
            def one_hot_enc(df,one_hot_enc_ft):
                one_hot_enc = OneHotEncoder(handle_unknown="ignore")
                one_hot_enc.fit(df[one_hot_enc_ft])
                # get the result of the one hot encoding columns names
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
                # change the array of the one hot encoding to a dataframe with the column names
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),columns=feat_names_one_hot_enc,index=df.index)
                return df
            # function to concatenat the one hot encoded features with the rest of features that were not encoded
            def concat_with_rest(df,one_hot_enc_df,one_hot_enc_ft):
                # get the rest of the features
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                # concatenate the rest of the features with the one hot encoded features
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]],axis=1)
                return df_concat
            # one hot encoded dataframe
            one_hot_enc_df = one_hot_enc(df,self.one_hot_enc_ft)
            # returns the concatenated dataframe
            full_df_one_hot_enc = concat_with_rest(df,one_hot_enc_df,self.one_hot_enc_ft)
            print(full_df_one_hot_enc.tail(25))
            return full_df_one_hot_enc  
        else:
            print("One or more features are not in the dataframe")
            return df


class OrdinalFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,ordinal_enc_ft = ['Education level']):
        self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Education level' in df.columns:
            ordinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print("Education level is not in the dataframe")
            return df

class MinMaxWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['Age', 'Income', 'Employment length']):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class ChangeToNumTarget(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Is high risk' in df.columns:
            df['Is high risk'] = pd.to_numeric(df['Is high risk'])
            return df
        else:
            print("Is high risk is not in the dataframe")
            return df

class OversampleSMOTE(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Is high risk' in df.columns:
            # SMOTE function to oversample the minority class to fix the imbalance data
            smote = SMOTE()
            X_bal, y_bal = smote.fit_resample(df.iloc[:,:-1],df.iloc[:,-1])
            X_y_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return X_y_bal
        else:
            print("Is high risk is not in the dataframe")
            return df


def full_pipeline(df):
    # Create the pipeline that will call all the class from OutlierRemoval to OversampleSMOTE in one go
    pipeline = Pipeline([
        ('outlier_remover', OutlierRemover()),
        ('feature_dropper', DropFeatures()),
        ('time_conversion_handler', TimeConversionHandler()),
        ('retiree_handler', RetireeHandler()),
        ('skewness_handler', SkewnessHandler()),
        ('binning_num_to_yn', BinningNumToYN()),
        ('one_hot_with_feat_names', OneHotWithFeatNames()),
        ('ordinal_feat_names', OrdinalFeatNames()),
        ('min_max_with_feat_names', MinMaxWithFeatNames()),
        ('change_to_num_target', ChangeToNumTarget()),
        ('oversample_smote', OversampleSMOTE())
    ])
    df_pipe_prep = pipeline.fit_transform(df)
    return df_pipe_prep

print(len(train_copy))


# Save the entire pipeline (including transformers and model)
pipeline_model_path = "pipeline_model.joblib"


# Function to detect fraud (for simulation purposes)
def detect_fraud():
    st.write("""
    # Assessment
    """)

    #Gender input
    st.write("""
    ## Gender
    """)
    input_gender = st.radio('Select you gender',['Male','Female'], index=0)


    # Age input slider
    st.write("""
    ## Age
    """)
    input_age = np.negative(st.slider('Select your age', value=42, min_value=18, max_value=70, step=1) * 365.25)




    # Marital status input dropdown
    st.write("""
    ## Marital status
    """)
    marital_status_values = list(value_cnt_norm_cal(full_data,'Marital status').index)
    marital_status_key = ['Married', 'Single/not married', 'Civil marriage', 'Separated', 'Widowed']
    marital_status_dict = dict(zip(marital_status_key,marital_status_values))
    input_marital_status_key = st.selectbox('Select your marital status', marital_status_key)
    input_marital_status_val = marital_status_dict.get(input_marital_status_key)


    # Family member count
    st.write("""
    ## Family member count
    """)
    fam_member_count = float(st.selectbox('Select your family member count', [1,2,3,4,5,6]))


    # Dwelling type dropdown
    st.write("""
    ## Dwelling type
    """)
    dwelling_type_values = list(value_cnt_norm_cal(full_data,'Dwelling').index)
    dwelling_type_key = ['House / apartment', 'Live with parents', 'Municipal apartment ', 'Rented apartment', 'Office apartment', 'Co-op apartment']
    dwelling_type_dict = dict(zip(dwelling_type_key,dwelling_type_values))
    input_dwelling_type_key = st.selectbox('Select the type of dwelling you reside in', dwelling_type_key)
    input_dwelling_type_val = dwelling_type_dict.get(input_dwelling_type_key)


    # Income
    st.write("""
    ## Income
    """)
    input_income = int(st.text_input('Enter your income (in USD)',0))


    # Employment status dropdown
    st.write("""
    ## Employment status
    """)
    employment_status_values = list(value_cnt_norm_cal(full_data,'Employment status').index)
    employment_status_key = ['Working','Commercial associate','Pensioner','State servant','Student']
    employment_status_dict = dict(zip(employment_status_key,employment_status_values))
    input_employment_status_key = st.selectbox('Select your employment status', employment_status_key)
    input_employment_status_val = employment_status_dict.get(input_employment_status_key)


    # Employment length input slider
    st.write("""
    ## Employment length
    """)
    input_employment_length = np.negative(st.slider('Select your employment length', value=6, min_value=0, max_value=30, step=1) * 365.25)


    # Education level dropdown
    st.write("""
    ## Education level
    """)
    edu_level_values = list(value_cnt_norm_cal(full_data,'Education level').index)
    edu_level_key = ['Secondary school','Higher education','Incomplete higher','Lower secondary','Academic degree']
    edu_level_dict = dict(zip(edu_level_key,edu_level_values))
    input_edu_level_key = st.selectbox('Select your education status', edu_level_key)
    input_edu_level_val = edu_level_dict.get(input_edu_level_key)


    # Car ownship input
    st.write("""
    ## Car ownship
    """)
    input_car_ownship = st.radio('Do you own a car?',['Yes','No'], index=0)

    # Property ownship input
    st.write("""
    ## Property ownship
    """)
    input_prop_ownship = st.radio('Do you own a property?',['Yes','No'], index=0)


    # Work phone input
    st.write("""
    ## Work phone
    """)
    input_work_phone = st.radio('Do you have a work phone?',['Yes','No'], index=0)
    work_phone_dict = {'Yes':1,'No':0}
    work_phone_val = work_phone_dict.get(input_work_phone)

    # Phone input
    st.write("""
    ## Phone
    """)
    input_phone = st.radio('Do you have a phone?',['Yes','No'], index=0)
    work_dict = {'Yes':1,'No':0}
    phone_val = work_dict.get(input_phone)

    # Email input
    st.write("""
    ## Email
    """)
    input_email = st.radio('Do you have an email?',['Yes','No'], index=0)
    email_dict = {'Yes':1,'No':0}
    email_val = email_dict.get(input_email)

    st.markdown('##')
    st.markdown('##')
    

    profile_to_predict = [0, # ID
                    input_gender[:1], # gender
                    input_car_ownship[:1], # car ownership
                    input_prop_ownship[:1], # property ownership
                    0, # Children count (which will be dropped in the pipeline)
                    input_income, # Income
                    input_employment_status_val, # Employment status
                    input_edu_level_val, # Education level
                    input_marital_status_val, # Marital status
                    input_dwelling_type_val, # Dwelling type
                    input_age, # Age
                    input_employment_length,    # Employment length
                    phone_val, # Has a mobile phone (which will be dropped in the pipeline)
                    work_phone_val, # Work phone
                    phone_val, # Phone
                    email_val,  # Email
                    None, # Job title (which will be dropped in the pipeline)
                    fam_member_count,  # Family member count
                    0.00, # Account age (which will be dropped in the pipeline)
                    1 # target set to 0 as a placeholder
                    ]



   

    loaded_pipeline = joblib.load(pipeline_model_path)

    profile_to_predict_df = pd.DataFrame([profile_to_predict],columns=test_copy.columns)

    # add the profile to predict as a last row in the train data
    test_copy_with_profile_to_pred = pd.concat([test_copy,profile_to_predict_df],ignore_index=True)

    # whole dataset prepared
    test_copy_with_profile_to_pred_prep = full_pipeline(test_copy_with_profile_to_pred)

    # Get the row with the ID = 0, and drop the ID, and target(placeholder) column
    profile_to_pred_prep = test_copy_with_profile_to_pred_prep.drop(['Is high risk'], axis = 1)
    
    # Use the loaded model for prediction
    # Button
    if st.button('Predict'):
        
        prediction = loaded_pipeline.predict(profile_to_pred_prep)
        print(prediction)
        if prediction[0]==1 or (input_income < 5000 or (input_car_ownship[:1] == 0 and input_prop_ownship[:1] == 0)):
            st.warning("Fraud Detected: This application has a high likelihood of fraud.")
        else:
            st.success("No Fraud Detected: The application is likely legitimate.")

# Function to simulate Stock Market Analysis
def simulate_stock_market_analysis():
    st.markdown("<div class='header'>Stock Market Analysis</div>", unsafe_allow_html=True)

    # Load sample stock data using yfinance
    ticker_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2022-01-01"))

    # Download historical stock data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Display stock data
    st.write(stock_data.head())

    # Fetch stock data
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META']
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    st.write('Downloading stock data...')
    for stock in tech_list:
        globals()[stock] = yf.download(stock, start, end)

    # Display summary statistics
    st.write('Summary Statistics for AAPL:')
    st.write(AAPL.describe())

    # Plot historical view of closing prices
    st.write('Historical View of Closing Prices:')
    plt.figure(figsize=(15, 10))
    for i, company in enumerate([AAPL, GOOG, MSFT, AMZN], 1):
        plt.subplot(2, 2, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {tech_list[i - 1]}")
    st.pyplot()

    # ... (previous code)

    # Plot Moving Averages
    st.write('Moving Averages:')
    ma_day = [10, 20, 50]
    for ma in ma_day:
        for company in [AAPL, GOOG, MSFT, AMZN]:
            column_name = f"MA for {ma} days"
            company[column_name] = company['Adj Close'].rolling(ma).mean()

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0, 0])
    axes[0, 0].set_title('APPLE')

    GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
    axes[0,1].set_title('GOOGLE')

    MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
    axes[1,0].set_title('MICROSOFT')

    AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
    axes[1,1].set_title('AMAZON')

    fig.tight_layout()

    st.pyplot()

    closing_df = yf.download(tech_list, start=start, end=end)['Adj Close']

    # Make a new tech returns DataFrame
    tech_rets = closing_df.pct_change()
    tech_rets.head()

    # Correlation Analysis
    st.write('Correlation Analysis:')
    sns.pairplot(tech_rets, kind='reg')
    st.pyplot()

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock return')

    plt.subplot(2, 2, 2)
    sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock closing price')
    st.pyplot()

    rets = tech_rets.dropna()

    area = np.pi * 20

    plt.figure(figsize=(10, 8))
    plt.scatter(rets.mean(), rets.std(), s=area)
    plt.xlabel('Expected return')
    plt.ylabel('Risk')

    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                     arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

    st.pyplot(plt)

    # LSTM Model Prediction
    st.write('LSTM Model Prediction:')

    # Use the stock symbols directly
    data = closing_df[tech_list].copy()
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95))

    # Scale the data
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    
    yf.pdr_override()

    # For time stamps
    # The tech stocks we'll use for this analysis
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN','META']

    # Set up End and Start times for data grab
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN','META']

    
    # Load sample stock data using yfinance

    ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start = st.date_input("Select Start Date", pd.to_datetime("2012-01-01"))
    end = st.date_input("Select End Date", pd.to_datetime("2023-11-20"))

    for stock in tech_list:
        globals()[stock] = yf.download(stock, start, end)

    company_list = [AAPL, GOOG, MSFT, AMZN]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON","META"]

    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name

    df = pd.concat(company_list, axis=0)
    df.tail(10)

    # Summary Stats
    AAPL.describe()

    # Let's see a historical view of the closing price
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {tech_list[i - 1]}")

    plt.tight_layout()

    ma_day = [10, 20, 50]

    for ma in ma_day:
        for company in company_list:
            column_name = f"MA for {ma} days"
            company[column_name] = company['Adj Close'].rolling(ma).mean()

    # Grab all the closing prices for the tech stock list into one DataFrame

    closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']

    # Make a new tech returns DataFrame
    tech_rets = closing_df.pct_change()
    tech_rets.head()

    # We can simply call pairplot on our DataFrame for an automatic visual analysis 
    # of all the comparisons

    sns.pairplot(tech_rets, kind='reg')

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock return')

    plt.subplot(2, 2, 2)
    sns.heatmap(closing_df.corr(), annot=True, cmap='summer')
    plt.title('Correlation of stock closing price')

    rets = tech_rets.dropna()

    area = np.pi * 20

    plt.figure(figsize=(10, 8))
    plt.scatter(rets.mean(), rets.std(), s=area)
    plt.xlabel('Expected return')
    plt.ylabel('Risk')

    for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                     arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

    # Get the stock quote
    df = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
    # Show the data
    df

    plt.figure(figsize=(16,6))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()

    # Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape

   

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    st.line_chart(valid)

    # Show the valid and predicted prices
    st.write(valid)

def main():
    # Sidebar
    st.sidebar.markdown("<div class='menu' style='background-color:#0066ff'>Menu</div>", unsafe_allow_html=True)
    option = st.sidebar.selectbox("Select an option", ("Credit Risk Assessment", "Stock Market Analysis"))

    if option == "Credit Risk Assessment":
        simulate_credit_risk_assessment()

    elif option == "Stock Market Analysis":
        simulate_stock_market_analysis()

if __name__ == "__main__":
    main()


