import tensorflow as tf
from tensorflow import keras
import numpy as  np

imdb = keras.datasets.imdb
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
#该数据集已经经过预处理，每个样本都是一个整数数组，表示影评中的字词。每个标签都是整数值0/1，表示负面/正面
# print("Training entries:{},labels:{}".format(len(train_data),len(train_labels)))
# print(train_data[0])

#将整数转换回文本
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
# print(decode_review(train_data[0]))

#准备数据
#影评（整数数组）必须转换为张量，才能馈送到神经网络中
#由于影评的长度必须相同，我们将使用pad_sequences函数将长度标准化：
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# print(len(train_data[0]),len(train_data[1]))
# print(train_data[0])

#构建模型
#在本示例中，输入数据由字词-索引数组构成。要预测的标签是0/1
#input shape is the vocabulary count used for the movie reviews(10000words)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#创建验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#每一epochs都进行F1计算
import numpy as np
from keras.callbacks import Callback
from keras.engine.training import Model
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
class Metrics(Callback):
    def on_train_begin(self,logs={}):
        self.val_fls=[]
        self.val_recalls = []
        self.val_precisions=[]

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')
        self.val_fls.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))
        return


metrics = Metrics()
from keras.callbacks import EarlyStopping
earlystopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=8, verbose=0, mode='max')
#训练模型
#用有512个样本的小批次训练模型40个周期。
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=90,
                    batch_size=512,
                    validation_data=(x_val,y_val),
                    callbacks=[metrics, earlystopping],
                    verbose=1)


#评估模型
results = model.evaluate(test_data,test_labels)
print(results)

#创建准确率和损失随时间变化的图
history_dict = history.history
history_dict.keys()
#dict_keys(['loss','val_loss','val_acc','acc'])

import  matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
# "bo"is for "blue dot"
plt.plot(epochs,loss,'bo',label='Training loss')
#b is for "solid blue line"
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b',marker='x', linestyle='dashed',
        linewidth=0.5, markersize=5, label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')#loss
plt.legend()

plt.show()

