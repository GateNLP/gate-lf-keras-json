import sys
sys.path.append("/Users/xingyi/Gate/learningFrameWorkDeepLearning/gate-lf-python-data/")
sys.path.append('/Users/xingyi/Gate/learningFrameWorkDeepLearning/gate-lf-keras-json/')
from gatelfdata import Dataset
from gatelfkerasjson import KerasWrapperImpl1
import json
metaFile = sys.argv[1]
modelSavePrefix = sys.argv[2]

singleData= "[1.0,0.0,0.8471,0.13533,0.73638,-0.06151,0.87873,0.0826,0.88928,-0.09139,0.78735,0.06678,0.80668,-0.00351,0.79262,-0.01054,0.85764,-0.04569,0.8717,-0.03515,0.81722,-0.0949,0.71002,0.04394,0.86467,-0.15114,0.81147,-0.04822,0.78207,-0.00703,0.75747,-0.06678,0.85764,-0.06151]"
instancedata = json.loads(singleData)
print(instancedata)
ds = Dataset(metaFile)

kerasModel = KerasWrapperImpl1(ds)
kerasModel.loadModel(modelSavePrefix)
label, prediction = kerasModel.applyModel(instancedata)
print(label, prediction)
