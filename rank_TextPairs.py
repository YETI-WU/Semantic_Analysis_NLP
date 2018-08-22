import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, Conv1D, Dropout, GlobalMaxPooling1D, Dense, Dot
from keras import regularizers
from keras.layers.merge import Concatenate 
from keras.models import Model




def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]      # number of training examples
    X_indices = np.zeros((m, max_len))
    for i in range(m):  # loop over training examples
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j+1
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1            
    emb_dim = word_to_vec_map["cucumber"].shape[0]  
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def rank_TextPairs_CNN(Doc, Que):
    ##### Sentence Model #####
    # Document
    maxLength_Doc = len(max(Doc, key=len).split())
    input_Doc = Input(shape = (maxLength_Doc,))
    #Doc = Embedding(input_dim=ques_vocab_size, output_dim=50, input_length=maxLength_Doc, weights=[embedding], trainable=False)(input_Doc)
    Doc_indeces = sentences_to_indices(Doc, word_to_index, maxLength_Doc)
    Doc = pretrained_embedding_layer(word_to_vec_map, Doc_indeces)
    Doc = Conv1D(filters=100, kernel_size=5, strides=1, padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(1e-5))(Doc)
    Doc = Dropout(0.5)(Doc)
    Doc = GlobalMaxPooling1D()(Doc)
    # Query
    maxLength_Que = len(max(Que, key=len).split())
    input_Que = Input(shape = (maxLength_Que,))
    #Que = Embedding(input_dim=ques_vocab_size, output_dim=50, input_length=maxLength_Que, weights=[embedding], trainable=False)(input_Que)
    Que_indeces = sentences_to_indices(Que, word_to_index, maxLength_Que)
    Que = pretrained_embedding_layer(word_to_vec_map, Que_indeces)
    Que = Conv1D(filters=100, kernel_size=5, strides=1, padding='same', activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(1e-5))(Que)
    Que = Dropout(0.5)(Que)
    Que = GlobalMaxPooling1D()(Que)
    
    ##### Matching Text Pairs #####
    # Similarity Matching
    MX_d = Dense(units=100)(Doc)
    Sim = Dot([Que,MX_d], axes=-1)
    # Join Layer
    Match = Concatenate([Doc, Sim, Que])
    # Hidden Layer
    Match = Dense(units=201,activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(1e-4))(Match)
    Match = Dropout(0.5)(Match)
    # Softmax
    Match = Dense(units=2, activation='softmax')(Match)
    
    model = Model(inputs=[input_Doc,input_Que], outputs=Match)
    
    return model

if __name__ == '__main__':
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_train = dict()
    Doc = newsgroups_train['data']
    Que = newsgroups_train['target']
 

    # Build Model Graph
    model = rank_TextPairs_CNN(Doc, Que) 
    model.summary()

    # Compile Model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) # use 'adam' optimizer, different than paper

    
    # Train model 
    train_history = model.fit(Doc, Que, epochs = 25, batch_size = 32) # need negative samples for training
    #print(train_history.history.keys())
    
    # Plot history of loss
    plt.plot(train_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_history_loss'], loc='upper right')
    plt.show()
    
    # plot history of acc
    plt.plot(train_history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_history_acc'], loc='lower right')
    plt.show()
