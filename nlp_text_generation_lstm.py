from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras import Model

import numpy as np
import random
import io


# código original: https://keras.io/examples/generative/lstm_character_level_text_generation/

path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
text = text.replace('\n', ' ')
print(f'Corpus length: {len(text)}')

chars = sorted(list(set(text)))
print('Total chars: {len(chars}')
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# extraemos secuencias de longitud 'maxlen' cada 3 carácteres
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:(i + maxlen)])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

# usamos one-hot encoding para cada frase. Una frase consiste en 40 (maxlen)
# vectores de ceros (de longitud len(chars)) con un uno en la posoción correspondiente al caracter
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

inputs = Input(shape=(maxlen, len(chars)))
lstm_output = LSTM(units=128, return_sequences=True)(inputs)
lstm_output = LSTM(units=128)(lstm_output)
output = Dense(units=len(chars), activation='softmax')(lstm_output)
model = Model(inputs, output)

optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # elige un elemento dado un vector de probabilidades (preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


epochs = 40
batch_size = 128

for epoch in range(epochs):
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print()
    print(f'Generating text after epoch: {epoch + 1}')

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('...Diversity:', diversity)

        generated = ''
        sentence = text[start_index:(start_index + maxlen)]
        print(f'...Generating with seed: "{sentence}"')

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_char_index = sample(preds, diversity)
            next_char = indices_char[next_char_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print('...Generated: ', generated)
        print()

# output of last training epoch
"""
Generating text after epoch: 40
...Diversity: 0.2
...Generating with seed: "ne wants to see--oneself--brought him to"
...Generated:  o the strone and beco not a sent of therer and and the promsed the the such of the subtranges a late and 
the sense at the senses to the sub a solo the content of the sense, and the super the so sonting the same then and the 
so mover the moral to the moral to ans the comple to him of the moral the consurent in the sec(ize of which the and man 
a sore the super the moral the the same then the subien 

...Diversity: 0.5
...Generating with seed: "ne wants to see--oneself--brought him to"
...Generated:  o hes or bin concerloined, and the of pain salg there self is alustin appropray not belatnes to his orect 
of late and the it in all a bictut a bose the whercesice the the master.   120  =the costenting or tempt with or relient 
in his ment of the past abso stile is in the sact of s other the sact of the will in prove to did so centur the parte, 
and the pase the the moral rief and ones to the moral 

...Diversity: 1.0
...Generating with seed: "ne wants to see--oneself--brought him to"
...Generated:  elalo phase. when forclie eccurioidedeste noh the risian catepisting of sus subys alfe--counsustr famars 
fard that meriviss--with otterd and "thegiouid he it justs in resivadity than he inthle cark, the chand that muple 
hally as feimdæ in for,--thing, hes u was to  asm iral elhateds eurces) the truthce moft his misinrce to the conseque 
aw innok theselv is is tbuncly, gfor wfindt the prespests the 

...Diversity: 1.2
...Generating with seed: "ne wants to see--oneself--brought him to"
...Generated:  o ded ohide and morals liximooy ranore, weced-wingruldy provient: to who . throng toorsne? crustt leud 
soperboudads event! ciccurinonly osk, wheses the ake betlee, inuencely; the bops acpeses i course thante whichs 
ucnalanly lefomeo-man hefyer; hein thenyed, finttenp moralite be ups: forlmment, ger feardt and prevosts, 
cstucinice--thic intreuvtes, anibw woot beykoius appereonau ons prtd" forwy, ha


"""