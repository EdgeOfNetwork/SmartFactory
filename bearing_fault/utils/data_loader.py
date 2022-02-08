import pandas as pd
import os
import numpy as np



# class DATALOADER:
#   def __init__(self, sth):
#     self.sth = sth
#
#   def data_loader():
#     cwru_12khz()
#     path = "../CWRU/"
#     df = pd.read_csv(path + "")

#여기를 기준으로 import 해서 쏴주는것일텐데

def cwru_12khz(path):
  matlab_files_name = {}
  # Normal
  matlab_files_name["Normal_0"] = "../input/cwru-bearing-dataset-mat/97.mat"
  matlab_files_name["Normal_1"] = "../input/cwru-bearing-dataset-mat/98.mat"
  matlab_files_name["Normal_2"] = "../input/cwru-bearing-dataset-mat/99.mat"
  matlab_files_name["Normal_3"] = "../input/cwru-bearing-dataset-mat/100.mat"
  # DE Inner Race 0.007 inches
  matlab_files_name["DEIR007_0"] = "../input/cwru-bearing-dataset-mat/105.mat"
  matlab_files_name["DEIR007_1"] = "../input/cwru-bearing-dataset-mat/106.mat"
  matlab_files_name["DEIR007_2"] = "../input/cwru-bearing-dataset-mat/107.mat"
  matlab_files_name["DEIR007_3"] = "../input/cwru-bearing-dataset-mat/108.mat"
  # DE Inner Race 0.014 inches
  matlab_files_name["DEIR014_0"] = "../input/cwru-bearing-dataset-mat/169.mat"
  matlab_files_name["DEIR014_1"] = "../input/cwru-bearing-dataset-mat/170.mat"
  matlab_files_name["DEIR014_2"] = "../input/cwru-bearing-dataset-mat/171.mat"
  matlab_files_name["DEIR014_3"] = "../input/cwru-bearing-dataset-mat/172.mat"
  # DE Inner Race 0.021 inches
  matlab_files_name["DEIR021_0"] = "../input/cwru-bearing-dataset-mat/209.mat"
  matlab_files_name["DEIR021_1"] = "../input/cwru-bearing-dataset-mat/210.mat"
  matlab_files_name["DEIR021_2"] = "../input/cwru-bearing-dataset-mat/211.mat"
  matlab_files_name["DEIR021_3"] = "../input/cwru-bearing-dataset-mat/212.mat"
  # DE Inner Race 0.028 inches
  matlab_files_name["DEIR028_0"] = "../input/cwru-bearing-dataset-mat/3001.mat"
  matlab_files_name["DEIR028_1"] = "../input/cwru-bearing-dataset-mat/3002.mat"
  matlab_files_name["DEIR028_2"] = "../input/cwru-bearing-dataset-mat/3003.mat"
  matlab_files_name["DEIR028_3"] = "../input/cwru-bearing-dataset-mat/3004.mat"

  # DE Ball 0.007 inches
  matlab_files_name["DEB007_0"] = "../input/cwru-bearing-dataset-mat/118.mat"
  matlab_files_name["DEB007_1"] = "../input/cwru-bearing-dataset-mat/119.mat"
  matlab_files_name["DEB007_2"] = "../input/cwru-bearing-dataset-mat/120.mat"
  matlab_files_name["DEB007_3"] = "../input/cwru-bearing-dataset-mat/121.mat"
  # DE Ball 0.014 inches
  matlab_files_name["DEB014_0"] = "../input/cwru-bearing-dataset-mat/185.mat"
  matlab_files_name["DEB014_1"] = "../input/cwru-bearing-dataset-mat/186.mat"
  matlab_files_name["DEB014_2"] = "../input/cwru-bearing-dataset-mat/187.mat"
  matlab_files_name["DEB014_3"] = "../input/cwru-bearing-dataset-mat/188.mat"
  # DE Ball 0.021 inches
  matlab_files_name["DEB021_0"] = "../input/cwru-bearing-dataset-mat/222.mat"
  matlab_files_name["DEB021_1"] = "../input/cwru-bearing-dataset-mat/223.mat"
  matlab_files_name["DEB021_2"] = "../input/cwru-bearing-dataset-mat/224.mat"
  matlab_files_name["DEB021_3"] = "../input/cwru-bearing-dataset-mat/225.mat"
  # DE Ball 0.028 inches
  matlab_files_name["DEB028_0"] = "../input/cwru-bearing-dataset-mat/3005.mat"
  matlab_files_name["DEB028_1"] = "../input/cwru-bearing-dataset-mat/3006.mat"
  matlab_files_name["DEB028_2"] = "../input/cwru-bearing-dataset-mat/3007.mat"
  matlab_files_name["DEB028_3"] = "../input/cwru-bearing-dataset-mat/3008.mat"

  # DE Outer race 0.007 inches centered @6:00
  matlab_files_name["DEOR@6_007_0"] = "../input/cwru-bearing-dataset-mat/130.mat"
  matlab_files_name["DEOR@6_007_1"] = "../input/cwru-bearing-dataset-mat/131.mat"
  matlab_files_name["DEOR@6_007_2"] = "../input/cwru-bearing-dataset-mat/132.mat"
  matlab_files_name["DEOR@6_007_3"] = "../input/cwru-bearing-dataset-mat/133.mat"
  # DE Outer race 0.014 inches centered @6:00
  matlab_files_name["DEOR@6_014_0"] = "../input/cwru-bearing-dataset-mat/197.mat"
  matlab_files_name["DEOR@6_014_1"] = "../input/cwru-bearing-dataset-mat/198.mat"
  matlab_files_name["DEOR@6_014_2"] = "../input/cwru-bearing-dataset-mat/199.mat"
  matlab_files_name["DEOR@6_014_3"] = "../input/cwru-bearing-dataset-mat/200.mat"
  # DE Outer race 0.021 inches centered @6:00
  matlab_files_name["DEOR@6_021_0"] = "../input/cwru-bearing-dataset-mat/234.mat"
  matlab_files_name["DEOR@6_021_1"] = "../input/cwru-bearing-dataset-mat/235.mat"
  matlab_files_name["DEOR@6_021_2"] = "../input/cwru-bearing-dataset-mat/236.mat"
  matlab_files_name["DEOR@6_021_3"] = "../input/cwru-bearing-dataset-mat/237.mat"

  # DE Outer race 0.007 inches centered @3:00
  matlab_files_name["DEOR@3_007_0"] = "../input/cwru-bearing-dataset-mat/144.mat"
  matlab_files_name["DEOR@3_007_1"] = "../input/cwru-bearing-dataset-mat/145.mat"
  matlab_files_name["DEOR@3_007_2"] = "../input/cwru-bearing-dataset-mat/146.mat"
  matlab_files_name["DEOR@3_007_3"] = "../input/cwru-bearing-dataset-mat/147.mat"
  # DE Outer race 0.021 inches centered @3:00
  matlab_files_name["DEOR@3_021_0"] = "../input/cwru-bearing-dataset-mat/246.mat"
  matlab_files_name["DEOR@3_021_1"] = "../input/cwru-bearing-dataset-mat/247.mat"
  matlab_files_name["DEOR@3_021_2"] = "../input/cwru-bearing-dataset-mat/248.mat"
  matlab_files_name["DEOR@3_021_3"] = "../input/cwru-bearing-dataset-mat/249.mat"

  # DE Outer race 0.007 inches centered @12:00
  matlab_files_name["DEOR@12_007_0"] = "../input/cwru-bearing-dataset-mat/156.mat"
  matlab_files_name["DEOR@12_007_1"] = "../input/cwru-bearing-dataset-mat/158.mat"
  matlab_files_name["DEOR@12_007_2"] = "../input/cwru-bearing-dataset-mat/159.mat"
  matlab_files_name["DEOR@12_007_3"] = "../input/cwru-bearing-dataset-mat/160.mat"
  # DE Outer race 0.021 inches centered @12:00
  matlab_files_name["DEOR@12_021_0"] = "../input/cwru-bearing-dataset-mat/258.mat"
  matlab_files_name["DEOR@12_021_1"] = "../input/cwru-bearing-dataset-mat/259.mat"
  matlab_files_name["DEOR@12_021_2"] = "../input/cwru-bearing-dataset-mat/260.mat"
  matlab_files_name["DEOR@12_021_3"] = "../input/cwru-bearing-dataset-mat/261.mat"
  return matlab_files_name

import tensorflow
from tensorflow.keras.datasets import fashion_mnist

def data_loader_fashionmnist():
  return fashion_mnist.load_data()

def readucr(filename):
  data = np.loadtxt(filename, delimiter="\t")
  y = data[:, 0]
  x = data[:, 1:]
  return x, y.astype(int)
