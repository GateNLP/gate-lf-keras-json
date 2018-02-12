import sys
sys.path.append("/Users/xingyi/Gate/learningFrameWorkDeepLearning/gate-lf-python-data/")
sys.path.append('/Users/xingyi/PycharmProjects/gate-lf-keras-wrapper')
from gatelfdata import Dataset
from kerasWrapper import KerasWrapperImpl1
metaFile = sys.argv[1]

ds = Dataset(metaFile)

kerasModel = KerasWrapperImpl1(ds)
kerasModel.genKerasModel()
kerasModel.trainModel()


