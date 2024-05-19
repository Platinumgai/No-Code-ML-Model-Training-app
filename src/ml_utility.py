import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Getting the Working Directory of the utility.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# getting the parent directory
parent_dir = os.path.dirname(working_dir)


def read_data(file_name):
    file_path = f'{parent_dir}/data/{file_name}'
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        return df


def preprocess_data(df, target_column, scaler_type):
    # split a features and target
    x = df.drop(columns=[target_column])
    y = df[target_column]

    # Checking if there are only numerical or Categorical columns
    numerical_cols = x.select_dtypes(include=['number']).columns
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns

    if len(numerical_cols) == 0:
        pass
    else:
        # split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Inpute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        x_train[numerical_cols] = num_imputer.fit_transform(x_train[numerical_cols])
        x_test[numerical_cols] = num_imputer.fit_transform(x_test[numerical_cols])

        # Scaler the numerical features based on scaler type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()

        x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
        x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])


    if len(categorical_cols) == 0:
        pass
    else:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        x_train[categorical_cols] = cat_imputer.fit_transform(x_train[categorical_cols])
        x_test[categorical_cols] = cat_imputer.fit_transform(x_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder()
        x_train_encoded = encoder.fit_transform(x_train[categorical_cols])
        x_test_encoded = encoder.fit_transform(x_test[categorical_cols])
        x_train_encoded = pd.DataFrame(x_train_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        x_test_encoded = pd.DataFrame(x_test_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        x_train = pd.concat([x_train.drop(columns=categorical_cols), x_train_encoded], axis=1)
        x_test = pd.concat([x_test.drop(columns=categorical_cols), x_test_encoded], axis=1)

    return x_train, x_test, y_train, y_test


# Train the model
def train_model(x_train, y_train, model, model_name):
    # training the selected model
    model.fit(x_train, y_train)
    # Saaving the Trained Model
    with open(f'{parent_dir}/trained_model/{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)
    return model


# Evaluate the model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    return accuracy

