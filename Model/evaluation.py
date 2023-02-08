from tensorflow.keras.models import load_model
from attention import AttentionLayer
model = load_model("summarymodel.h5", custom_objects={'AttentionLayer': AttentionLayer})
encoder_model = load_model("encoder_model.h5", custom_objects={'AttentionLayer': AttentionLayer})
decoder_model = load_model("decoder_model.h5", custom_objects={'AttentionLayer': AttentionLayer})
import re
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional, \
    AbstractRNNCell, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
from attention import AttentionLayer
# Data cleaning
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
max_text_len = 30
max_summary_len = 8
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}

class newtextsummary():


    def __init__(self,txt):
        self.txt=txt
        super(newtextsummary, self).__init__()
        # tf.config.run_functions_eagerly(True)

    def dataclean(self):
        def text_cleaner(text, num):
            newString = text.lower()
            newString = BeautifulSoup(newString, "lxml").text
            newString = re.sub(r'\([^)]*\)', '', newString)
            newString = re.sub('"', '', newString)
            newString = ' '.join(
                [contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
            newString = re.sub(r"'s\b", "", newString)
            newString = re.sub("[^a-zA-Z]", " ", newString)
            newString = re.sub('[m]{2,}', 'mm', newString)
            print(newString)
            if (num == 0):
                tokens = [w for w in newString.split() if not w in stop_words]
            else:
                tokens = newString.split()
            long_words = []
            for i in tokens:
                if len(i) > 1:  # removing short word
                    long_words.append(i)
            return (" ".join(long_words)).strip()

        # call the function
        cleaned_text = []
        df = pd.DataFrame({"text":[self.txt]})

        for t in df['text']:
            print(t)
            cleaned_text.append(text_cleaner(t, 0))
        print(cleaned_text[:5])
        data = pd.DataFrame()
        # call the function
        data['cleaned_text'] = cleaned_text
        data.replace('', np.nan, inplace=True)
        data.dropna(axis=0, inplace=True)
        max_text_len = 30
        max_summary_len = 8

        # changing new dataframe with cleaned data  for text and summary
        cleaned_text = np.array(data['cleaned_text'])
        # cleaned_summary = np.array(data['cleaned_summary'])

        short_text = []
        # short_summary = []
        print("cleaned text", cleaned_text)
        for i in range(len(cleaned_text)):
            if len(cleaned_text[i].split()) <= max_text_len:
                short_text.append(cleaned_text[i].lower())

        data = pd.DataFrame({'text': short_text})
        x_tr = data['text']
        print("x_tr",x_tr)
        # prepare a tokenizer for reviews on training data
        x_tokenizer = Tokenizer()
        x_tokenizer.fit_on_texts(list(x_tr))
        thresh = 4
        cnt = 0
        tot_cnt = 0
        freq = 0
        tot_freq = 0

        for key, value in x_tokenizer.word_counts.items():
            tot_cnt = tot_cnt + 1
            tot_freq = tot_freq + value
            if (value < thresh):
                cnt = cnt + 1
                freq = freq + value

        print("% of rare words in vocabulary:", (cnt / tot_cnt) * 100)
        print("Total Coverage of rare words:", (freq / tot_freq) * 100)

        # prepare a tokenizer for reviews on training data - text
        x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
        x_tokenizer.fit_on_texts(list(x_tr))

        # convert text sequences into integer sequences
        x_tr_seq = x_tokenizer.texts_to_sequences(x_tr)

        # padding zero upto maximum length
        x_tr = pad_sequences(x_tr_seq, maxlen=max_text_len, padding='post')
        print("x_tr with padding",x_tr)
        # size of vocabulary ( +1 for padding token)
        x_voc = x_tokenizer.num_words + 1
        # print(x_voc)

        return x_tr

    def encoder_model(self):

        # inference for prediction purpose
        # Encode the input sequence to get the feature vector
        encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs)
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h,
                                                                                     decoder_state_input_c])

        # attention inference
        attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_inf_concat)

        # Final decoder model
        decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])
        return decoder_model

    def decode_sequence(self,input_seq):
        with open('reverse_target_word_index.json') as f:
            data = f.read()
        reverse_target_word_index = dict(json.loads(data))
        f.close()

        with open('target_word_index.json') as f:
            data = f.read()
        target_word_index = dict(json.loads(data))
        f.close()

        # Encode the input as state vectors.
        e_out, e_h, e_c = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))

        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = target_word_index['sostok']

        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:

            output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            print(sampled_token_index)
            sampled_token = reverse_target_word_index[str(sampled_token_index)]

            if (sampled_token != 'eostok'):
                decoded_sentence += ' ' + sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_summary_len - 1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence

    def model_prediction(self):

        x_tr=self.dataclean()
        print("x tr shape",x_tr.shape)
        ycap = self.decode_sequence(x_tr)
        return ycap

#x = newtextsummary('I had been amazon restaurent , liked biryani a lot ,ultimate taste and reasonable price')
x = newtextsummary("i AM OKAY WITH COFFEE , NOT AT ! RIGHT TIME !!")
#x = newtextsummary("Transfer learning is a popular method in computer vision because it allows us to build accurate models in a timesaving way (Rawat & Wang 2017). With transfer learning, instead of starting the learning process from scratch, you start from patterns that have been learned when solving a different problem. This way you leverage previous learnings and avoid starting from scratch. Take it as the deep learning version of Chartres’ expression ‘standing on the shoulder of giants’.In computer vision, transfer learning is usually expressed through the use of pre-trained models. A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. Accordingly, due to the computational cost of training such models, it is common practice to import and use models from published literature (e.g. VGG, Inception, MobileNet). A comprehensive review of pre-trained models’ performance on computer vision problems using data from the ImageNet (Deng et al. 2009) challenge is presented by Canziani et al. (2016).")
#x = newtextsummary("Transfer learning is a popular method in computer vision because it allows us to build accurate models")
resppred=x.model_prediction()

print("Summary",resppred)

