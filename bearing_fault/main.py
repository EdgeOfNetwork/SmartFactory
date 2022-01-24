from utils import data_loader
from models.model import AutoEncoder



def model_build(latent_dim):
    ae = AutoEncoder(latent_dim)

def preprocessing(x_train, x_test):
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    return x_train, x_test

if __name__ == "__main__":

    (x_train, _), (x_test, _ ) = data_loader.data_loader_fashionmnist() # data loading

    x_train, x_test = preprocessing(x_train, x_test) # preprocessing

    #set hyperparameters
    latent_dim = 64

    model_build(latent_dim)


    path = ""
    df = data_loader(path)
    print(df)
    #print("hi")
"""
TODO : 

DAE + DCNN 으로 검출해보기

0. CWRU 데이터 EDA 및 데이터 로딩 등 데이터 성질 파악 

TODO : 

[] 1D인 vibration 데이터 2D 변환 
[] 2D 진동 데이터에 Noise 넣어보기
[] Denoising 작업 해보기
[] 모델 학습 및 Denoising 된 결과물 데이터셋으로 저장해보기
------------------------------------------------------
*병신 SSA를 할꺼면 1D에다가 넣어야할 꺼 아냐
[] Denosing 데이터에 각자 전처리 씌워보고 따로 저장하기
[] FFT
[] Wavelet
[] SSA
------------------------------------------------------
[] 

1. DAE와 DCNN모델 CLASS화 하기
2. 인공지능 모델에 CWRU 데이터 적용
3. 성능 파악 및 시각화 요소 파악
4. 

 
"""