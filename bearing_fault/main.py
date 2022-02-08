from utils import data_loader
from models.model import AutoEncoder
import numpy as np
from models.transformer import build_model
from tensorflow import keras

# def model_build(latent_dim):
#     ae = AutoEncoder(latent_dim)
#
# def preprocessing(x_train, x_test):
#     x_train = x_train.astype("float32") / 255.
#     x_test = x_test.astype("float32") / 255.
#     return x_train, x_test

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

def ford_engine_classify():
    pass

if __name__ == "__main__":

    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test , y_test  = readucr(root_url + "FordA_TEST.tsv")

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test  = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    n_classes = len(np.unique(y_train))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    input_shape = x_train.shape[1:]

    model = build_model(n_classes, input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(x_train, y_train, validation_split=0.2, epochs=200, batch_size=64, callbacks=callbacks)

    model.evaluate(x_test, y_test, verbose=1)

    # (x_train, _), (x_test, _ ) = data_loader.data_loader_fashionmnist() # data loading
    # x_train, x_test = preprocessing(x_train, x_test) # preprocessing

    # set hyperparameters
    # latent_dim = 64
    # model_build(latent_dim)
    # path = ""
    # df = data_loader(path)
    # print(df)
    # #print("hi")
"""
TODO : 

1. Applying SSA transformation to bearing data set
 
[] keras-fold engine 예제 모듈화해보기  
[] -> transformer 구축
=======================================================
[] bearing dataset에 preprocessing 해보기
[] Denoising 작업 해보기
[] 모델 학습 및 Denoising 된 결과물 데이터셋으로 저장해보기
------------------------------------------------------
[] Denosing 데이터에 각자 전처리 씌워보고 따로 저장하기
[] FFT
[] Wavelet
[] SSA
------------------------------------------------------
"""