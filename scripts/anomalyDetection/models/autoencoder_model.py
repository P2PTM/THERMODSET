import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scripts.anomalyDetection.config import config
from scripts.anomalyDetection.utils.logger import logger
from scripts.anomalyDetection.models.base_model import BaseModel


class AutoencoderDetector(BaseModel):
    def __init__(self):
        super().__init__("Autoencoder")
        self.model = None
        self.best_params = None
        self.threshold = None

    def _create_model(self, input_dim, encoding_dims, dropout_rate, activation):
        model = Sequential()

        # Encoder
        model.add(Dense(encoding_dims[0], activation=activation, input_shape=(input_dim,)))
        model.add(Dropout(dropout_rate))

        for dim in encoding_dims[1:]:
            model.add(Dense(dim, activation=activation))
            model.add(Dropout(dropout_rate))

        # Decoder (symmetric to encoder)
        for dim in reversed(encoding_dims[:-1]):
            model.add(Dense(dim, activation=activation))
            model.add(Dropout(dropout_rate))

        model.add(Dense(input_dim, activation='linear'))

        return model

    def train(self, X_train):
        logger.info("Training Autoencoder model...")
        input_dim = X_train.shape[1]
        best_loss = float('inf')
        best_config = {}

        for encoding_dims in config.MODELS.autoencoder_params['encoding_dims']:
            for dropout_rate in config.MODELS.autoencoder_params['dropout_rate']:
                for learning_rate in config.MODELS.autoencoder_params['learning_rate']:
                    for batch_size in config.MODELS.autoencoder_params['batch_size']:
                        for epochs in config.MODELS.autoencoder_params['epochs']:
                            model = self._create_model(
                                input_dim,
                                encoding_dims,
                                dropout_rate,
                                'relu'
                            )

                            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

                            early_stopping = EarlyStopping(
                                monitor='val_loss',
                                patience=10,
                                restore_best_weights=True
                            )

                            history = model.fit(
                                X_train, X_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=[early_stopping],
                                verbose=0
                            )

                            val_loss = min(history.history['val_loss'])

                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_config = {
                                    'encoding_dims': encoding_dims,
                                    'dropout_rate': dropout_rate,
                                    'learning_rate': learning_rate,
                                    'batch_size': batch_size,
                                    'epochs': epochs
                                }

        # Retrain with best configuration
        self.model = self._create_model(
            input_dim,
            best_config['encoding_dims'],
            best_config['dropout_rate'],
            'relu'
        )

        self.model.compile(optimizer=Adam(learning_rate=best_config['learning_rate']), loss='mean_squared_error')

        self.model.fit(
            X_train, X_train,
            epochs=best_config['epochs'],
            batch_size=best_config['batch_size'],
            validation_split=0.2,
            verbose=0
        )

        self.best_params = best_config
        logger.info(f"Best parameters: {self.best_params}")

        # Compute reconstruction error threshold
        reconstructed = self.model.predict(X_train)
        reconstruction_errors = np.mean(np.power(X_train - reconstructed, 2), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        reconstructed = self.model.predict(X)
        reconstruction_errors = np.mean(np.power(X - reconstructed, 2), axis=1)

        # Binary predictions based on threshold
        y_pred = (reconstruction_errors > self.threshold).astype(int)

        return y_pred, reconstruction_errors

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        # Compute the reconstruction errors
        reconstructed = self.model.predict(X)
        reconstruction_errors = np.mean(np.power(X - reconstructed, 2), axis=1)

        # Normalize reconstruction errors to [0, 1] for probability interpretation
        min_error = reconstruction_errors.min()
        max_error = reconstruction_errors.max()
        proba = (reconstruction_errors - min_error) / (max_error - min_error)

        return np.vstack([1 - proba, proba]).T
