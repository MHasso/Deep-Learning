from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from data_loader import DataLoader
from keras_model import KerasModel

def main():

    filename = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

    #load data
    dataloader = DataLoader(filename)
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    dataloader.load(columns)
    dataloader.label_encode('income')

    #feature engineering apply OneHotEncoder to categorical variables and MinMaxScaler to continous variables
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    continuous_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    dataloader.preprocess(categorical_cols, continuous_cols)
    
    dataloader.split()
    y_pred_baseline = dataloader.baseline_mode()
    
    #print the rog score
    print(roc_auc_score(dataloader.y_test, y_pred_baseline))

    kerasmodel = KerasModel()
    kerasmodel.create_classifier()
    kerasmodel.create_pipeline(dataloader.preprocessor)
    kerasmodel.fit(dataloader.X_train, dataloader.y_train)

    # Does the model predict better than random?
    print(dataloader.X_test.shape)
    y_prob = kerasmodel.pipeline.predict_proba(dataloader.X_test)[:, 1]
    print(dataloader.X_test, y_prob)
    auc_score = roc_auc_score(dataloader.y_test, y_prob)
    print(auc_score)

    # Generate an ROC curve for your model.
    RocCurveDisplay.from_estimator(kerasmodel.pipeline, dataloader.X_test, y=dataloader.y_test)



if __name__ == "__main__":
    main()



