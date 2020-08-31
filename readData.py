import pandas as pd
import re
from datetime import datetime
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping

SEQUENCE_LEN = 10
BATCH_SIZE = 32

def convertLogToDataframe(file):
    data = []

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n\n')

        for line in lines:
            try:
                time_entered = line.split('|')[0].strip()
                time_entered = datetime.strptime(time_entered, '%Y-%m-%d_%H:%M:%S')
                message = line.split('|')[1:]
                message = '|'.join(message).strip()
                username, channel, message = re.search(':(.*)!.*@.*\.tmi\.twitch\.tv PRIVMSG #(.*) :(.*)',message).groups()

                d = {
                    'dt':time_entered,
                    'channel':channel,
                    'username':username,
                    'message':message
                }

                data.append(d)
            except Exception:
                pass
    return pd.DataFrame().from_records(data)

def generator(sentenceList, nextWordList, batchSize):
    index = 0
    while True:
        x = np.zeros((batchSize, SEQUENCE_LEN), dtype = np.int)
        y = np.zeros((batchSize),dtype=np.int)
        for i in range(batchSize):
            for t, w in enumerate(sentenceList[index % len(sentenceList)]):
                x[i, t] = word_indices[w]
            #print("v1 " + str(nextWordList[index % len(sentenceList)]))
            #print("made it this far" + str(word_indices[nextWordList[index % len(sentenceList)]]))
            y[i] = word_indices[nextWordList[index % len(sentenceList)]]
            index += 1
        yield x, y

df = convertLogToDataframe('chat.log')

print(df.head)

cleanMessage = df['message'].map(lambda x: x.lower() if isinstance(x,str) else x)
#print(cleanMessage)
wordDictionary = {}

for s in range(len(cleanMessage)):
    words = cleanMessage[s].split(' ')

    for word in words:
        wordDictionary[word] = wordDictionary.get(word,0) + 1


#print(wordDictionary)

test = pd.DataFrame.from_dict(wordDictionary,orient='index', columns= ['Frequency'])
test = test[test['Frequency'] > 5]
test.reset_index(inplace=True)

word_indices = {c:i for i,c in enumerate(test['index'])}
indices_word = {i:c for i,c in enumerate(test['index'])}

#print(len(word_indices))
#print(indices_word)
#print(test.head())
#print(test[test['Frequency'] > 5])

sequences = []
next_words = []
ignored = 0

for i in range(0, len(cleanMessage)):
    cleanWords = cleanMessage[i].split(' ')
    for j in range(len(cleanWords) - SEQUENCE_LEN):
        if all(w in word_indices for w in cleanWords[j:j+SEQUENCE_LEN +1]):
            sequences.append(cleanWords[j:j+SEQUENCE_LEN])
            next_words.append(cleanWords[j+SEQUENCE_LEN])
            #print('word sequence allowed:', cleanWords[j:j + SEQUENCE_LEN])
        else:
            #print('word sequence not allowed:', cleanWords[j:j + SEQUENCE_LEN])
            ignored += 1
            #time.sleep(5)

print("Number of ignored:", ignored)
print("Remaining sequences:", len(sequences))

def buildModel(dropout = 0.2):
    print('Building model..')
    model = Sequential()
    model.add(Embedding(input_dim=len(word_indices), output_dim=1024))
    model.add(Bidirectional(LSTM(128)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(word_indices)))
    model.add(Activation('softmax'))
    return model


def shuffleTrainSet(sequences, nextWords):
    tmpSequences = []
    tmpNextWord = []

    for i in np.random.permutation(len(sequences)):
        tmpSequences.append(sequences[i])
        tmpNextWord.append(nextWords[i])
        
    return tmpSequences, tmpNextWord

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    #This function is run at the end of each epoch
    with open("examples.txt", "w", encoding= 'utf-8') as examplesFile:
        examplesFile.write('\n---Generating text after Epoch: %d\n' %epoch)

        seed_index = np.random.randint((len(xTrain)))
        seed = xTrain[seed_index]

        for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
            sentence = seed
            examplesFile.write('----Diversity:' + str(diversity) + '\n')
            examplesFile.write('----Generating with seed:\n"' + ' '.join(sentence) + '"\n')

            for i in range(50):
                x_pred = np.zeros((1, SEQUENCE_LEN))
                for t, word in enumerate(sentence):
                    x_pred[0,t] = word_indices[word]
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = indices_word[next_index]

                sentence = sentence[1:]
                sentence.append(next_word)

                examplesFile.write(" "+next_word)
            examplesFile.write('\n')
        examplesFile.write('='*80 + '\n')


xTrain, yTrain = shuffleTrainSet(sequences, next_words)

model = buildModel()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                "loss{loss:.4f}-acc{acc:.4f}" % \
                (len(word_indices), SEQUENCE_LEN, 5)

checkpoint = ModelCheckpoint(file_path,monitor='acc', save_best_only=True)
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='acc', patience=20)
callbacks_list = [checkpoint, print_callback, early_stopping]

examplesFile = open("examples.txt", "w")
model.fit_generator(generator(xTrain, yTrain, BATCH_SIZE),
                    steps_per_epoch=int(len(xTrain)/BATCH_SIZE)+1,
                    epochs=100,
                    callbacks=callbacks_list)




