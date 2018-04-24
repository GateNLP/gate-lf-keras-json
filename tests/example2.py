import sys
sys.path.append("/Users/xingyi/Gate/learningFrameWorkDeepLearning/gate-lf-python-data/")
sys.path.append('/Users/xingyi/Gate/learningFrameWorkDeepLearning/gate-lf-keras-json/')
from gatelfdata import Dataset
from gatelfkerasjson import KerasWrapperImpl1
metaFile = sys.argv[1]
modelSavePrefix = sys.argv[2]
ds = Dataset(metaFile)

kerasModel = KerasWrapperImpl1(ds)
kerasModel.genKerasModel()
kerasModel.trainModel()
kerasModel.saveModel(modelSavePrefix)

