import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
from keras.layers import LSTM, Conv1D, Flatten, Dropout, Merge, TimeDistributed, MaxPooling1D, Conv2D
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical

class KerasWrapperImpl1(object):
    def __init__(self, dataSet, configDict=None):
        assert isinstance(dataSet, object)
        self.ds = dataSet
        self.ds.target.set_as_onehot(True)

        self.inputMask = [None] * self.ds.nFeatures
        self.featureKindList = [None] * self.ds.nFeatures
        #self.num_idxs = dataSet.get_numeric_feature_idxs()
        #self.nom_idxs = dataSet.get_nominal_feature_idxs()
        #self.ngr_idxs = dataSet.get_ngram_feature_idxs()
        self.num_idxs = dataSet.get_float_feature_idxs()
        self.nom_idxs = dataSet.get_index_feature_idxs()
        self.ngr_idxs = dataSet.get_indexlist_feature_idxs()

        self.genMask()
        self.inputMask = np.array(self.inputMask)
        print(self.inputMask)
        self.featureKindList = np.array(self.featureKindList)
        self.uniqueAttri, self.AttriCount = np.unique(self.inputMask, return_counts=True)
        self.embeddingSize = 16
        self.inputLayerList = []
        self.outputLayersList = []
        self.featureState = []
        self.model = None

    def genMask(self):
        maskIdDict = {}
        maskIdx = -1
        if len(self.num_idxs) > 0:
            maskIdx += 1
            maskIdDict['A'] = maskIdx
            for idx in self.num_idxs:
                self.inputMask[idx] = maskIdx
                self.featureKindList[idx] = 'A'

        if len(self.nom_idxs) > 0:
            for idx in self.nom_idxs:
                currentFeature = self.ds.features.features[idx]
                currentFeatureKind = currentFeature.attrinfo['code']
                attriId = currentFeature.attrinfo['featureId']
                if attriId not in maskIdDict:
                    maskIdx += 1
                    self.inputMask[idx] = maskIdx
                    maskIdDict[attriId] = maskIdx
                else:
                    self.inputMask[idx] = maskIdDict[attriId]
                self.featureKindList[idx] = currentFeatureKind

        if len(self.ngr_idxs) > 0:
            for idx in self.ngr_idxs:
                currentFeature = self.ds.features.features[idx]
                currentFeatureKind = currentFeature.attrinfo['code']
                attriId = currentFeature.attrinfo['featureId']
                if attriId not in maskIdDict:
                    maskIdx += 1
                    self.inputMask[idx] = maskIdx
                    maskIdDict[attriId] = maskIdx
                else:
                    self.inputMask[idx] = maskIdDict[attriId]
                self.featureKindList[idx] = currentFeatureKind

    def genKerasModel(self):
        if self.ds.isSequence:
            pass
        else:
            self.genInputLayer()
            self.genHiddenLayers()
            print(len(self.outputLayersList))
            if len(self.inputLayerList) > 1:
                allHidden = Concatenate()(self.outputLayersList)
            else:
                allHidden = self.outputLayersList[0]
            if self.ds.nClasses > 2:
                output = Dense(self.ds.nClasses, activation="softmax")(allHidden)
            else:
                # leave it for testing
                output = Dense(self.ds.nClasses, activation="sigmoid")(allHidden)
                # output = Dense(1, activation="sigmoid")(allHidden)
            model = Model(self.inputLayerList, output)
            model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
            model.summary()
        self.model = model

    def genHiddenLayers(self):
        for i in range(len(self.inputLayerList)):
            sequenceFeature = self.featureState[i][0]
            numFeature = self.featureState[i][1]
            current_output = self.outputLayersList[i]
            print(current_output)
            if numFeature:
                current_output = Dense(32)(current_output)
            else:
                if sequenceFeature:
                    current_output = LSTM(units=16, return_sequences=False)(current_output)
                    #current_output = Conv1D(filters=32, kernel_size=2, strides=1, activation='relu')(current_output)

                else:
                    current_output = Conv1D(filters=32, kernel_size=2, strides=1, activation='relu')(current_output)
                # current_output = MaxPooling1D(pool_size=(2))(current_output)
                    current_output = Dropout(0.2)(current_output)
                    current_output = Flatten()(current_output)
                    current_output = Dense(32)(current_output)
            self.outputLayersList[i] = current_output

    def genInputLayer(self):
        # unique, counts = np.unique(self.inputMask, return_counts=True)
        print(self.uniqueAttri)
        sequenceFeature = False
        numFeature = False
        for attribId in self.uniqueAttri:
            featureIndexList = np.where(self.inputMask == attribId)
            currentKindList = self.featureKindList[featureIndexList]
            if ('L' in currentKindList):
                vocabSize = len(self.ds.features.features[featureIndexList[0][0]].vocab.freqs) + 10000
                current_input = Input(shape=(len(currentKindList),))
                current_output = Embedding(vocabSize, self.embeddingSize, input_length=len(currentKindList))(
                    current_input)
            elif ('N' in currentKindList):
                sequenceFeature = True
                vocabSize = len(self.ds.features.features[featureIndexList[0][0]].vocab.freqs) + 10000
                current_input = Input(shape=(None,))

                current_output = Embedding(vocabSize, self.embeddingSize)(current_input)
            elif ('A' in currentKindList):
                numFeature = True
                current_input = Input(shape=(len(currentKindList),))
                current_output = Dense(10)(current_input)
            else:
                print('unsupported feature type')
            self.inputLayerList.append(current_input)
            self.outputLayersList.append(current_output)
            self.featureState.append([sequenceFeature, numFeature])

    def trainModel(self, batchSize=1, nb_epoch=5):
        self.ds.split(convert=True, keep_orig=False, validation_part=0.05)
        valset = self.ds.validation_set_converted(as_batch=True)
        valx = self.convertX(valset[0])
        valy = to_categorical(valset[1], num_classes=self.nClasses)
        newvalx = []
        for item in valx:
            newvalx.append(np.array(item))

        for i in range(nb_epoch):
            print('epoch ', i)
            self.trainKerasModelBatch(batchSize)
            print(newvalx[0].shape)
            valLoss = self.model.evaluate(x=newvalx, y=np.array(valy))
            print('val loss ', valLoss)

    def trainKerasModelBatch(self, batchSize):
        convertedTraining = self.ds.batches_converted(train=True, batch_size=batchSize)
        for batchInstances in convertedTraining:
            featureList = batchInstances[0]
            target = batchInstances[1]
            # print(featureList)
            miniBatchY = to_categorical(target,num_classes=self.nClasses)
            miniBatchX = self.convertX(featureList)
            # print(miniBatchY)
            # print(miniBatchX)
            self.trainMiniBatch(miniBatchX, miniBatchY)

    def convertX(self, xList):
        # numInputAttribute = max(self.inputMask)+1
        numInputAttribute = len(self.uniqueAttri)
        miniBatchX = [[] for i in range(numInputAttribute)]

        for xid in range(len(xList[0])):
            for eachAttribute in miniBatchX:
                eachAttribute.append([])
            for maskIdx in range(len(self.inputMask)):
                miniBatchX[self.inputMask[maskIdx]][-1].append(xList[maskIdx][xid])
        return miniBatchX

    def trainMiniBatch(self, miniBatchX, miniBatchY):
        # print('training')
        newX = []
        for item in miniBatchX:
            newX.append(np.array(item))
        # print(newX)
        # print(newX[0].shape)
        trainLoss = self.model.train_on_batch(x=newX, y=np.array(miniBatchY))
        # loss = self.model.test_on_batch(x=newX, y=np.array(miniBatchY))
        # print(trainLoss)
