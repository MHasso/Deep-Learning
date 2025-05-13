
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler




class DataLoader:

    def __init__(self, filepath):
        self.filepath = filepath

    def load(self, columns):
        self.df = pd.read_csv(self.filepath, header=None, names=columns, na_values='?', skipinitialspace=True)
    
    def clean(self):
        self.df.dropna(inplace=True)
    
    def label_encode(self, column):
        label_encoder = LabelEncoder()
        self.df[column] = label_encoder.fit_transform(self.df[column])

    def split(self):
        X = self.df.drop('income', axis=1)
        y = self.df['income']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    def baseline_mode(self):
        majority_class = self.y_train.mode()[0]
        return [majority_class] * len(self.y_test)
    
    def preprocess(self, categorical_column, continous_column):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_column),  
                ('num', MinMaxScaler(), continous_column)  
             ])
        self.preprocessor.fit_transform(self.df)
