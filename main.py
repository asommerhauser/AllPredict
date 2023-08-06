import os
import csv
import tensorflow as tf
import numpy as np
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.utils import pad_sequences

input_vocab = {}
output_vocab = {}

#***************** --DIAGNOSTICS-- *****************
def testShots(parsed, game):
    pointsParsed = 0
    points = 0
    for x in parsed:
        element = decodeInt(x[12])
        if element == 'shot':
            if decodeInt(x[15]) == 'made':
                if decodeInt(x[14]) == '2pt':
                    pointsParsed += 2
                elif decodeInt(x[14]) == '3pt':
                    pointsParsed += 3
                else:
                    pointsParsed += 1
    for x in game:
        if x['event_type'] == 'shot' or x['event_type'] == 'free throw':
            if x['result'] == 'made':
                points += int(x['points'])
    print(points, pointsParsed)
    if points == pointsParsed:
        return True
    else:
        return "POINTS ERROR"

#----------------------------------------------
#fillStep function:
#This function is used to fill the beginning of a timestep with all the constant information that will
#always be listed at the beginning of a timestep
#input: step - the individual timestep in question, a dictionary
#output: timestep - a list containing the string information that begins every timestep
#----------------------------------------------
def fillStep(step):
    timestep = []
    timestep.append(step['h1'])
    timestep.append(step['h2'])
    timestep.append(step['h3'])
    timestep.append(step['h4'])
    timestep.append(step['h5'])
    timestep.append(step['a1'])
    timestep.append(step['a2'])
    timestep.append(step['a3'])
    timestep.append(step['a4'])
    timestep.append(step['a5'])
    timestep.append(step['period'])
    timestep.append(step['remaining_time'][-5:])
    return timestep

#----------------------------------------------
#assistStep function:
#This function returns a properly formatted assist
#input: step - the shot step that is being turned into an assist
#output: timestep - a properly formatted assist step
#----------------------------------------------
def assistStep(step):
    timestep = fillStep(step)
    timestep.append('assist')
    timestep.append(step['assist'])
    if step['type'][:3] == '3pt':
        timestep.append('3pt')
    else:
        timestep.append('2pt')
    timestep.append('shot')
    return timestep

#----------------------------------------------
#shotStep function:
#This function returns a properly formatted shot
#input: step - the shot step that is being cleaned
#output: timestep - a properly formatted shot step
#----------------------------------------------
def shotStep(step):
    timestep = fillStep(step)
    timestep.append('shot')
    timestep.append(step['player'])
    if step['type'][:3] == '3pt':
        timestep.append('3pt')
    else:
        timestep.append('2pt')
    # Check for shot block
    if step['block'] != '':
        timestep.append('block')
    else:
        timestep.append(step['result'])
    return timestep

#----------------------------------------------
#assistStep function:
#This function returns a properly formatted block
#input: step - the shot step that is being turned into an block
#       next - the next event type that occurs in the game
#output: timestep - a properly formatted block step
#----------------------------------------------
def blockStep(step, next):
    timestep = fillStep(step)
    timestep.append('block')
    timestep.append(step['block'])
    timestep.append('block')
    timestep.append(next)
    return timestep

#----------------------------------------------
#reboundStep function:
#This function takes in a rebound event and returns a list of 16 items
#input: step - a rebound event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def reboundStep(step):
    timestep = fillStep(step)
    timestep.append('rebound')
    if step['type'] == 'team rebound':
        timestep.append('null')
        timestep.append(step['type'][8:])
    else:
        timestep.append(step['player'])
        timestep.append(step['type'][8:])
    if step['type'][8:] == 'defensive':
        timestep.append('cop')
    else:
        timestep.append('null')
    return timestep

#----------------------------------------------
#freeThrowStep function:
#This function takes in a free throw event and returns a list of 16 items
#input: step - a free throw event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def freeThrowStep(step):
    timestep = fillStep(step)
    timestep.append('shot')
    timestep.append(step['player'])
    timestep.append('free throw')
    timestep.append(step['result'])
    return timestep

#----------------------------------------------
#stealStep function:
#This function takes in a turnover event and returns a list of 16 items with a steal event
#input: step - a turnover event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def stealStep(step):
    timestep = fillStep(step)
    timestep.append('steal')
    timestep.append(step['steal'])
    timestep.append(step['player'])
    timestep.append('cop')
    return timestep

#----------------------------------------------
#turnoverStep function:
#This function takes in a turnover event and returns a list of 16 items with a turnover event
#input: step - a turnover event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def turnoverStep(step):
    #List of possible violations
    checkVio = ['3-second violation', 'shot clock', '8-second violation', 'lane violation', 'offensive goaltending',
                'palming', 'backcourt', '5-second violation', 'double dribble', 'discontinue dribble', 'illegal assist',
                'jump ball violation', 'offensive foul', 'illegal screen', 'basket from below', 'punched ball',
                'too many players', 'traveling', 'kicked ball']
    #List of possible errors
    checkError = ['lost ball', 'out of bounds lost ball', 'step out of bounds', 'bad pass', 'inbound']
    timestep = fillStep(step)
    timestep.append('turnover')
    if step['player'] == '':
        timestep.append('null')
    else:
        timestep.append(step['player'])
    if step['steal'] != '':
        timestep.append('steal')
    elif step['type'] == '':
        timestep.append('null')
    elif step['type'] in checkVio:
        timestep.append('violation')
    elif step['type'] in checkError:
        timestep.append('error')
    timestep.append('cop')
    return timestep

#----------------------------------------------
#foulStep function:
#This function takes in a foul event and returns a list of 16 items with a foul event
#input: step - a foul event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def foulStep(step, next):
    timestep = fillStep(step)
    timestep.append('foul')
    if step['event_type'] == 'foul':
        if step['player'] == '':
            timestep.append('team')
        else:
            timestep.append(step['player'])
        if step['type'] == '':
            timestep.append('null')
        elif step['type'][-9:] == 'technical':
            timestep.append('technical')
        else:
            timestep.append(step['type'])
        if next == 'ejection':
            timestep.append('ejection')
        else:
            timestep.append('foul')
    else:
        if step['type'] == 'coach technical foul':
            timestep.append('coach')
        elif step['player'] == '':
            timestep.append('null')
        else:
            timestep.append(step['player'])
        if step['type'] == 'coach technical foul':
            timestep.append('coach technical')
        elif step['type'] == 'defensive 3 seconds':
            timestep.append('defensive 3-second')
        else:
            timestep.append('double technical')
        if next == 'ejection':
            timestep.append('ejection')
        else:
            timestep.append('foul')
    return timestep

#----------------------------------------------
#subStep function:
#This function takes in a substitution event and returns a list of 16 items with a substitution event
#input: step - a substitution event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def subStep(step):
    timestep = fillStep(step)
    timestep.append('substitution')
    timestep.append(step['entered'])
    timestep.append(step['left'])
    timestep.append('substitution')
    return timestep

#----------------------------------------------
#tensorfy function:
#This function takes in a list of 16 items, encodes them and then inputs them into a tensor
#input: step - a list of 16 string items
#output: an encoded tensor of the 16 items
#----------------------------------------------
'''def tensorfy(step):
    count = 0
    for x in step:
        step[count] = encodeString(x)
        count += 1
    return tf.constant(step)'''

def tensorin(step):
    count = 0
    for x in step:
        step[count] = encodeInput(x)
        count += 1
    return tf.constant(step)

def tensorout(step):
    count = 0
    for x in step:
        step[count] = encodeOutput(x)
        count += 1
    return tf.constant(step)

#----------------------------------------------
#encodeString function:
#This function encodes strings and updates the vocab and
#input: the string to be encoded and the dictionary of all encoded terms
#output: dictionary of all encoded terms with one added
#----------------------------------------------
'''def encodeString(string):
    if string not in vocab:
        code = len(vocab) + 1
        vocab[string] = code
    return vocab[string]'''

def encodeInput(string):
    if string not in input_vocab:
        code = len(input_vocab) + 1
        input_vocab[string] = code
    return input_vocab[string]

def encodeOutput(string):
    if string not in output_vocab:
        code = len(output_vocab) + 1
        output_vocab[string] = code
    return output_vocab[string]

#----------------------------------------------
#decodeInt function:
#This function decodes integers
#input: encoded int
#output: decoded string
#----------------------------------------------
def decodeInt(integer):
    for string, value in output_vocab.items():
        if value == integer:
            return string
    return None

#----------------------------------------------
#padList function:
#This function adds null values onto the end of a game so that the tensor will always be the same shape
#input: list of any length less than 1000
#output: list of length 1000
#----------------------------------------------
'''def padList(steps):
    hold = encodeString('null')
    emptyStep = [hold, hold, hold, hold, hold, hold, hold, hold, hold, hold, hold, hold, hold, hold, hold, hold]
    for x in range(1000-len(steps)):
        steps.append(emptyStep)
    return steps'''

#----------------------------------------------
#scoreGame function:
#This function takes in a tensor output and determines which team scored more points
#input: tensor (formatted as the RNN output)
#output: bool (true = home win, false = away win
#----------------------------------------------
def scoreGame(game):
    home = 0
    away = 0
    x = 0
    while decodeInt(game[x][0]) != 'null':
        if decodeInt(game[x][15]) == 'made':
            points = 0
            pointer = decodeInt(game[x][14])
            if pointer == '2pt':
                points = 2
            elif pointer == '3pt':
                points = 3
            else:
                points = 1
            player = decodeInt(game[x][13])
            if player == decodeInt(game[x][0]) or player == decodeInt(game[x][1]) or player == decodeInt(game[x][2]) or player == decodeInt(game[x][3]) or player == decodeInt(game[x][4]):
                home += points
            else:
                away += points
        x+=1
    if home > away:
        return True
    else:
        return False

#----------------------------------------------
#parseGame function:
#This function is used to clean the raw basketball data into a format that is suitable for the RNN
#input: path - path to a csv file
#output: timesteps - full list of timesteps from the csv file
#----------------------------------------------
def parseGame(path):
    print(path)
    timesteps = []
    steps = -1
    home = []
    away = []
    input = []
    #Opens the file
    with open(path) as file:
        steps += 1
        dict = csv.DictReader(file)
        data = [row for row in dict]
        count = -1
        #input date
        input.append(data[0]['date'][-2:])
        input.append(data[0]['date'][5:7])
        input.append(data[0]['date'][:4])
        #input players
        for x in range(1, 6):
            string = 'h' + str(x)
            home.append(data[0][string])
        for x in range(1, 6):
            string = 'a' + str(x)
            away.append(data[0][string])
        #Loops through each event in the game
        for x in data:
            count += 1
            #EVENT CHECKING
            #shot
            if x['event_type'] == 'shot':
                blocked = False
                #check for if the shot was assisted
                if x['assist'] != '':
                    timesteps.append(assistStep(x))
                #Shot information
                timesteps.append(shotStep(x))
                #Check for block
                if x['block'] != '':
                    blocked = True
                #Shot block event
                if blocked == True:
                    timesteps.append(blockStep(x, data[count + 1]['event_type']))
            #free throw
            elif x['event_type'] == 'free throw':
                timesteps.append(freeThrowStep(x))
            #rebound
            elif x['event_type'] == 'rebound':
                timesteps.append(reboundStep(x))
            #turnover
            elif x['event_type'] == 'turnover':
                if x['type'] != 'no turnover':
                    timesteps.append(turnoverStep(x))
                #check for steal
                if x['steal'] != '':
                    timesteps.append(stealStep(x))
            #foul
            elif x['event_type'] == 'foul':
                timesteps.append(foulStep(x, data[count+1]['event_type']))
            elif x['event_type'] == 'technical foul':
                timesteps.append(foulStep(x, data[count+1]['event_type']))
            #substitution
            elif x['event_type'] == 'substitution':
                timesteps.append(subStep(x))
                for y in range(1, 6):
                    string = 'h' + str(y)
                    if x[string] not in home:
                        home.append(x[string])
                for y in range(1, 6):
                    string = 'a' + str(y)
                    if x[string] not in away:
                        away.append(x[string])
    for x in home:
        input.append(x)
    for x in range(15 - len(home)):
        input.append('div')
    for x in away:
        input.append(x)
    for x in range(15 - len(away)):
        input.append('div')
    timesteps_sequence = []
    timesteps_sequence.append('start')
    for step in timesteps:
        timesteps_sequence = timesteps_sequence + step
    timesteps_sequence.append('end')
    return (tensorin(input), tensorout(timesteps_sequence))

folderPath = r'C:\BallPredict'
input_data = []
output_data = []
count = 0
for filename in os.listdir(folderPath):
    data = {}
    path = folderPath + '\\' + filename
    hold = parseGame(path)
    input_data.append(hold[0])
    output_data.append(hold[1])
    if count > 8:
        break
    count += 1

# Vocabulary size and embedding dimension
input_vocab_size = len(input_vocab)+1
output_vocab_size = len(output_vocab)+1
embedding_dim = 128

# Model hyperparameters
hidden_units = 256

#Encoder
encoder_inputs = tf.keras.layers.Input(shape=(None,))
embedding_vector = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_rnn = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_rnn(embedding_vector)
encoder_states = [state_h, state_c]

#Decoder
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(output_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_rnn = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states)

#Attension mechanism
attention = tf.keras.layers.Attention()([decoder_outputs, encoder_outputs])
decoder_concat = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, attention])

#Output Layer
decoder_dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')
output_logits = decoder_dense(decoder_concat)

#Define the model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output_logits)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print(model.summary())

#Train the model
padded_input = pad_sequences(input_data, maxlen=35, padding='post')
padded_output = pad_sequences(output_data, maxlen=20000, padding='post')
#model.fit([padded_input, padded_output], padded_output, batch_size=32, epochs=10)

#----------------------------Inference----------------------------
#Encoder
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

#Decoder
decoder_state_input_h = tf.keras.layers.Input(shape=(hidden_units,))
decoder_state_input_c = tf.keras.layers.Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_rnn_outputs, state_h, state_c = decoder_rnn(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

#Attention
attention_inference = tf.keras.layers.Attention()([decoder_rnn_outputs, encoder_outputs])
decoder_concat_inference = tf.keras.layers.Concatenate(axis=-1)([decoder_rnn_outputs, attention_inference])

decoder_outputs = decoder_dense(decoder_concat_inference)

decoder_states_outputs = [tf.keras.layers.Input(shape=(hidden_units,)) for _ in decoder_states_inputs]

decoder_model = tf.keras.models.Model(
    inputs=[decoder_inputs] + decoder_states_inputs,
    outputs=[decoder_outputs] + decoder_states_outputs
)