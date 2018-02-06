import sys
sys.path.append("/Users/xingyi/Gate/learningFrameWorkDeepLearning/gate-lf-python-data/")
from gatelfdata import Dataset

class KerasWrapper(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.inputMask = []
        self.featureKindList = []
        for currentFeature in self.features.features:
            self.inputMask.append(currentFeature.attrinfo['featureId']) 
            self.featureKindList.append(currentFeature.attrinfo['code'])
        self.inputMask = np.array(self.inputMask)
        self.featureKindList = np.array(self.featureKindList)
        self.embeddingSize = 16
        self.inputLayerList = []
        self.outputLayersList = []
        self.featureState = []
        self.model = None

    def getVocabList(self):
        unique, counts = np.unique(self.inputMask, return_counts=True)
        for attribId in unique:
            featureIndexList = np.where(self.inputMask == attribId)
            self.vocabList.append([])
            for featureIndex in featureIndexList:
                currentFeatureName = self.features.features[featureIndex].fname
                for word in self.meta['featureStats'][currentFeatureName]['stringCounts']:
                    if word not in vocabList[attribId]:
                        self.vocabList[attribId].append(word)

    def getSingleVocabList(self, featureIndexList):
        vocoList = []
        for featureIndex in featureIndexList:
            currentFeatureName = self.features.features[featureIndex].fname
            for word in self.meta['featureStats'][currentFeatureName]['stringCounts']:
                if word not in vocabList[attribId]:
                    vocabList.append(word)
        return vocoList



    def genKerasModel(self):
        self.genInputLayer()
        self.genHiddenLayers()
        sequenceTarget = False
        if sequenceTarget:
            pass
        else:
            print(self.outputLayersList)
            allHidden = Concatenate()(self.outputLayersList)
            print(allHidden)
            numOutput = len(self.metaData['targetStats']['stringCounts'])
            print(numOutput)
            output = Dense(self.nClasses, activation = "softmax")(allHidden)
            model = Model(self.inputLayerList, output)
            model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
            model.summary()
        self.model= model


    def genInputLayer(self):
        unique, counts = np.unique(self.inputMask, return_counts=True)
        for attribId in unique:
            featureIndexList = np.where(self.inputMask == attribId)
            print(featureIndexList)
            currentKindList = self.featureKindList[featureIndexList]
            if ('N' not in currentKindList):
                isSequence=False
                isNum = False
                vocabSize = len(self.features.features[featureIndexList[0]].vocab.freqs)
                current_input = Input(shape=(len(currentKindList),))
                #vocoList = self.getSingleVocabList(featureIndexList)
                current_output = Embedding(vocabSize, self.embeddingSize, input_length=len(currentKindList))(current_input)
            else:
                current_input = Input(shape=(None,))
                isSequence=True
                if attribId in self.metaData['vocabulary']:
                    current_output = Embedding(vocabSize, self.embeddingSize)(current_input)
                    isNum=False
                else:
                    print('n-gram do not accept numbers input')
                    sys.exit(0)
            self.inputLayerList.append(current_input)
            self.outputLayersList.append(current_output)
            self.featureState.append([isSequence,isNum])


    def genHiddenLayers(self):
        for i in range(len(self.inputLayerList)):
            isSequence = self.featureState[i][0]
            isNum = self.featureState[i][1]
            current_output = self.outputLayersList[i]
            print(current_output)
            if isNum == False:
                if isSequence:
                    current_output = LSTM(units=16, return_sequences=True)(current_output)
                    current_output = Conv1D(filters=32,kernel_size=2, strides=1, activation='relu')(current_output)
                else:
                    current_output = Conv1D(filters=32,kernel_size=2, strides=1, activation='relu')(current_output)
                #current_output = MaxPooling1D(pool_size=(2))(current_output)
                current_output = Dropout(0.2)(current_output)
                current_output = Flatten()(current_output)
                current_output = Dense(32)(current_output)
            self.outputLayersList[i] = current_output

    def trainModel(self, batchSize = 256, nb_epoch=5):
        valset = self.convert_to_file()
        for i in range(nb_epoch):
            print('epoch ', i)
            trainKerasModelBatch(valset)

    def trainKerasModelBatch(self, valset, batchSize = 256):
        convertedTraining=self.batches_converted(batch_size=batchSize)
        for batchInstances in convertedTraining:
            featureList = batchInstances[0]
            target = batchInstances[1]
            miniBatchY = target
            miniBatchX = self.convertX(featureList)
            self.trainMiniBatch(miniBatchX, miniBatchY)

    def convertX(self,xList):
        numInputFeatures = max(self.inputMask)+1
        miniBatchX = [[] for i in range(numInputFeatures)]
        for x in xList:
            for eachFeature in miniBatchX:
                eachFeature.append([])
            for maskIdx in range(len(self.inputMask)):
                miniBatchX[self.inputMask[maskIdx]][-1].append(x[maskIdx])
        return miniBatchX
        
            

    def trainMiniBatch(self, miniBatchX, miniBatchY):
        print('training')
        newX = []
        for item in miniBatchX:
            newX.append(np.array(item))
        trainLoss = self.model.train_on_batch(x=newX, y=np.array(miniBatchY))
        loss = self.model.test_on_batch(x=newX, y=np.array(miniBatchY))
        print(trainLoss)




