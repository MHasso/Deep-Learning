from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline

def build_model():
        model = Sequential()
        model.add(Dense(108, input_dim = 108, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
        return model

class KerasModel:
    
        
    def create_classifier(self):
        self.keras_classifier = KerasClassifier(model=build_model, epochs=10, batch_size=32)

    def create_pipeline(self, preprocessor):
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', self.keras_classifier)
            ])
        
    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
