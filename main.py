import os
import csv
import tensorflow as tf
import numpy as np
from keras.layers import LSTM, Dense, Input, Embedding
from keras.models import Model
from keras.utils import pad_sequences

#player_vocab is a dictionary that holds numbered keys each representing an individual player
player_vocab = {0: 'start', 1: 'end', 2: 'null'}
players_reversed = {'start': 0, 'end': 1, 'null': 2}
#rosters is a dictionary that holds each unique unordered combination of players together on a roster and pairs them with an int
rosters = {}
#token_vocab is a dictionary the holds numbered keys each corresponding to string that makes up the cell
token_vocab = {
    1: 'start',
    2: 'end',
    3: 'null',
    4: 'cop'
}
#cell_vocab is a dictionary that holds numbered keys, each that represent a unique numerical value that corresponds to the cells
cell_vocab = {}
#season_vocab is a list that holds all the seasons that are a part of the data set and one for next season
season_vocab = []
reg_or_play = [0, 1]

#***************** --DIAGNOSTICS-- *****************
'''def testShots(parsed, game):
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
        return "POINTS ERROR"'''

'''-------------------------------------------
encodeListToInt function:
this function is used to encode a list of integers into a singular integer
input: list of integers
output: single unique integer
----------------------------------------------'''
def encodeListToInt(lst):
    prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    result = 1
    for i, num in enumerate(lst):
        if i < len(prime_numbers):
            result *= prime_numbers[i] ** num
    return result

'''----------------------------------------------
decodeIntToList function:
this function is used to decode a singular integer into a list of integers
input: single unique integer
output: list of integers
----------------------------------------------'''
def decodeIntToList(encoded_int):
    prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    decoded_list = []
    for prime in prime_numbers:
        count = 0
        while encoded_int % prime == 0:
            encoded_int //= prime
            count += 1
        decoded_list.append(count)
    return decoded_list

#----------------------------------------------
#fillStep function:
#This function is used to fill the beginning of a timestep with all the constant information that will
#always be listed at the beginning of a timestep
#input: step - the individual timestep in question, a dictionary
#output: timestep - a list containing the string information that begins every timestep
#----------------------------------------------
def fillStep(step):
    timestep = []
    home = []
    away = []
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
    timestep.append(int(step['period']))
    return timestep

#----------------------------------------------
#assistStep function:
#This function returns a properly formatted assist
#input: step - the shot step that is being turned into an assist
#output: timestep - a properly formatted assist step
#----------------------------------------------
def assistStep(step):
    timestep = fillStep(step)
    after_steps = ['assist', step['assist']]
    if step['type'][:3] == '3pt':
        after_steps.append('3pt')
    else:
        after_steps.append('2pt')
    after_steps.append('shot')
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#shotStep function:
#This function returns a properly formatted shot
#input: step - the shot step that is being cleaned
#output: timestep - a properly formatted shot step
#----------------------------------------------
def shotStep(step):
    timestep = fillStep(step)
    after_steps = ['shot', step['player']]
    if step['type'][:3] == '3pt':
        after_steps.append('3pt')
    else:
        after_steps.append('2pt')
    # Check for shot block
    if step['block'] != '':
        after_steps.append('block')
    else:
        after_steps.append(step['result'])
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#assistStep function:
#This function returns a properly formatted block
#input: step - the shot step that is being turned into an block
#       next - the next event type that occurs in the game
#output: timestep - a properly formatted block step
#----------------------------------------------
def blockStep(step, next):
    timestep = fillStep(step)
    after_steps = ['block', step['block'], 'block', next]
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#reboundStep function:
#This function takes in a rebound event and returns a list of 16 items
#input: step - a rebound event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def reboundStep(step):
    timestep = fillStep(step)
    after_steps = ['rebound']
    if step['type'] == 'team rebound' or step['player'] == '':
        after_steps.append('null')
        after_steps.append(step['type'][8:])
    else:
        after_steps.append(step['player'])
        after_steps.append(step['type'][8:])
    if step['type'][8:] == 'defensive':
        after_steps.append('cop')
    else:
        after_steps.append('null')
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#freeThrowStep function:
#This function takes in a free throw event and returns a list of 16 items
#input: step - a free throw event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def freeThrowStep(step):
    timestep = fillStep(step)
    after_steps = ['shot', step['player'], 'free throw', step['result']]
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#stealStep function:
#This function takes in a turnover event and returns a list of 16 items with a steal event
#input: step - a turnover event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def stealStep(step):
    timestep = fillStep(step)
    after_steps = ['steal', step['steal'], step['player'], 'cop']
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#turnoverStep function:
#This function takes in a turnover event and returns a list of 16 items with a turnover event
#input: step - a turnover event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def turnoverStep(step):
    #List of possible violations
    check_vio = ['3-second violation', 'shot clock', '8-second violation', 'lane violation', 'offensive goaltending',
                'palming', 'backcourt', '5-second violation', 'double dribble', 'discontinue dribble', 'illegal assist',
                'jump ball violation', 'offensive foul', 'illegal screen', 'basket from below', 'punched ball',
                'too many players', 'traveling', 'kicked ball']
    #List of possible errors
    check_error = ['lost ball', 'out of bounds lost ball', 'step out of bounds', 'bad pass', 'inbound']
    timestep = fillStep(step)
    after_steps = ['turnover']
    if step['player'] == '':
        after_steps.append('null')
    else:
        after_steps.append(step['player'])
    if step['steal'] != '':
        after_steps.append('steal')
    elif step['type'] == '':
        after_steps.append('null')
    elif step['type'] in check_vio:
        after_steps.append('violation')
    elif step['type'] in check_error:
        after_steps.append('error')
    after_steps.append('cop')
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#foulStep function:
#This function takes in a foul event and returns a list of 16 items with a foul event
#input: step - a foul event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def foulStep(step, next):
    timestep = fillStep(step)
    after_steps = ['foul']
    if step['event_type'] == 'foul':
        if step['player'] == '':
            after_steps.append('team')
        else:
            after_steps.append(step['player'])
        if step['type'] == '':
            after_steps.append('null')
        elif step['type'][-9:] == 'technical':
            after_steps.append('technical')
        else:
            after_steps.append(step['type'])
        if next == 'ejection':
            after_steps.append('ejection')
        else:
            after_steps.append('foul')
    else:
        if step['type'] == 'coach technical foul':
            after_steps.append('coach')
        elif step['player'] == '':
            after_steps.append('null')
        else:
            after_steps.append(step['player'])
        if step['type'] == 'coach technical foul':
            after_steps.append('coach technical')
        elif step['type'] == 'defensive 3 seconds':
            after_steps.append('defensive 3-second')
        else:
            after_steps.append('double technical')
        if next == 'ejection':
            after_steps.append('ejection')
        else:
            after_steps.append('foul')
    full_list = timestep + after_steps
    return full_list

#----------------------------------------------
#subStep function:
#This function takes in a substitution event and returns a list of 16 items with a substitution event
#input: step - a substitution event (dictionary)
#output: timestep - a 16 item list of strings
#----------------------------------------------
def subStep(step):
    timestep = fillStep(step)
    after_steps = ['substitution', step['entered'], step['left'], 'substitution']
    full_list = timestep + after_steps
    return full_list

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

#----------------------------------------------
#encodeString function:
#This function encodes strings and updates the vocab and
#input: the string to be encoded and the dictionary of all encoded terms
#output: dictionary of all encoded terms with one added
#----------------------------------------------
def encodePlayer(stri):
    if stri not in player_vocab.values():
        code = len(player_vocab) + 1
        player_vocab[code] = stri
        players_reversed[stri] = code
        return code
    else:
        return players_reversed[stri]

def encodeRoster(lst):
    num = encodeListToInt(lst)
    if num not in rosters:
        code = len(rosters) + 1
        rosters[num] = code
    return rosters[num]

def encodeTokens(lst):
    encoded_list = []
    for token in lst:
        if token not in token_vocab:
            code = len(token_vocab) + 1
            token_vocab[token] = code
        encoded_list.append(token_vocab[token])
    return encoded_list

def encodeCell(lst):
    cell_code = encodeListToInt(lst)
    if cell_code not in cell_vocab:
        code = len(cell_vocab) + 1
        cell_vocab[code] = cell_code
        cell_vocab[code] = cell_code
    return cell_vocab[cell_code]

#----------------------------------------------
#decodeInt function:
#This function decodes integers
#input: encoded int
#output: decoded string
#----------------------------------------------
'''def decodeInt(integer):
    for string, value in output_vocab.items():
        if value == integer:
            return string
    return None'''

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
'''def scoreGame(game):
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
        return False'''

#----------------------------------------------
#parseGame function:
#This function is used to clean the raw basketball data into a format that is suitable for the RNN
#input: path - path to a csv file
#output: timesteps - full list of timesteps from the csv file
#----------------------------------------------
def parseGame(path):
    print(path)
    playoff = 0
    season = 0
    player_step = []
    timesteps = []
    steps = -1
    home = []
    away = []
    #timesteps.append(['start', 'start', 'start', 'start', 'start', 'start', 'start'])
    player_step = [encodePlayer('start')]
    #Opens the file
    with open(path) as file:
        steps += 1
        dict = csv.DictReader(file)
        data = [row for row in dict]
        count = -1
        #input season
        if int(path[21:][:2]) > 9:
            season = int(path[16:][:4]) + 1
        else:
            season = int(path[16:][:4])
        if season not in season_vocab:
            season_vocab.append(season)
        #input game state
        if data[0]['data_set'][8] == 'P':
            playoff = 1
        #input players
        for x in range(1, 6):
            string = 'h' + str(x)
            home.append(encodePlayer(data[0][string]))
        for x in range(1, 6):
            string = 'a' + str(x)
            away.append(encodePlayer(data[0][string]))
        #Loops through each event in the game
        for x in data:
            count += 1
            #EVENT CHECKING
            #shot
            if x['event_type'] == 'shot':
                #check for if the shot was assisted
                if x['assist'] != '':
                    #timesteps.append(assistStep(x))
                    player_step.append(encodePlayer(assistStep(x)[12]))
                    #print(str(count) + ': ' + 'ASSIST -', player_vocab[player_step[len(player_step) - 1]])
                #Shot information
                #timesteps.append(shotStep(x))
                player_step.append(encodePlayer(shotStep(x)[12]))
                #print(str(count) + ': ' + 'SHOT -', player_vocab[player_step[len(player_step) - 1]])
                #Check for block
                if x['block'] != '':
                    # timesteps.append(blockStep(x, data[count + 1]['event_type']))
                    player_step.append(encodePlayer(blockStep(x, data[count + 1]['event_type'])[12]))
                    #print(str(count) + ': ' + 'BLOCK -', player_vocab[player_step[len(player_step) - 1]])
            #free throw
            elif x['event_type'] == 'free throw':
                #timesteps.append(freeThrowStep(x))
                player_step.append(encodePlayer(freeThrowStep(x)[12]))
                #print(str(count) + ': ' + 'FREE THROW -', player_vocab[player_step[len(player_step) - 1]])
            #rebound
            elif x['event_type'] == 'rebound':
                #timesteps.append(reboundStep(x))
                player_step.append(encodePlayer(reboundStep(x)[12]))
                #print(str(count) + ': ' + 'REBOUND -', player_vocab[player_step[len(player_step) - 1]])
            #turnover
            elif x['event_type'] == 'turnover':
                if x['type'] != 'no turnover':
                    #timesteps.append(turnoverStep(x))
                    player_step.append(encodePlayer(turnoverStep(x)[12]))
                    #print(str(count) + ': ' + 'TURNOVER -', player_vocab[player_step[len(player_step) - 1]])
                #check for steal
                if x['steal'] != '':
                    #timesteps.append(stealStep(x))
                    player_step.append(encodePlayer(stealStep(x)[12]))
                    #print(str(count) + ': ' + 'STEAL -', player_vocab[player_step[len(player_step) - 1]])
            #foul
            elif x['event_type'] == 'foul':
                #timesteps.append(foulStep(x, data[count+1]['event_type']))
                player_step.append(encodePlayer(foulStep(x, data[count + 1]['event_type'])[12]))
                #print(str(count) + ': ' + 'FOUL -', player_vocab[player_step[len(player_step) - 1]])
            elif x['event_type'] == 'technical foul':
                #timesteps.append(foulStep(x, data[count+1]['event_type']))
                player_step.append(encodePlayer(foulStep(x, data[count+1]['event_type'])[12]))
                #print(str(count) + ': ' + 'TECH -', player_vocab[player_step[len(player_step) - 1]])
            #substitution
            elif x['event_type'] == 'substitution':
                #timesteps.append(subStep(x))
                for y in range(1, 6):
                    string = 'h' + str(y)
                    if x[string] not in home:
                        home.append(encodePlayer(x[string]))
                for y in range(1, 6):
                    string = 'a' + str(y)
                    if x[string] not in away:
                        away.append(encodePlayer(x[string]))
                player_step.append(encodePlayer(subStep(x)[12]))
                #print(str(count) + ': ' + 'SUB -', player_vocab[player_step[len(player_step) - 1]])
    player_step.append(encodePlayer('end'))
    return [tf.constant(playoff), tf.constant(season), tf.constant(home), tf.constant(away), tf.constant(player_step)]

folderPath = r'C:\BallPredict'
input_season = []
input_playoffs = []
input_home = []
input_away = []
output_data = []
count = 0
for filename in os.listdir(folderPath):
    data = {}
    path = folderPath + '\\' + filename
    hold = parseGame(path)
    #input_playoffs.append(hold[0])
    #input_season.append(hold[1])
    #input_home.append(hold[2])
    #input_away.append(hold[3])
    #output_data.append(hold[4])
    if count > 8:
        break
    count += 1

season_vocab.append(season_vocab[len(season_vocab) - 1] + 1)

#input_season = np.array(input_season)
#input_playoffs = np.array(input_playoffs)
#input_home = np.array(input_home)
#input_away = np.array(input_away)
#output_data = np.array(output_data)

season_input = Input(shape=(1,))
playoff_input = Input(shape=(1,))
home_input = Input(shape=(None,))
away_input = Input(shape=(None,))
output_input = Input(shape=(None,))

playoff_embedding = Embedding(input_dim=2, output_dim=32)(playoff_input)
season_embedding = Embedding(input_dim=len(season_vocab), output_dim=32)(season_input)
roster_embedding_layer = Embedding(input_dim=len(player_vocab), output_dim=32)
home_embedding = roster_embedding_layer(home_input)
away_embedding = roster_embedding_layer(away_input)

model = Model(inputs=[season_input, playoff_input, home_input, away_input], outputs=output_input)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit([input_season, input_playoffs, input_home, input_away], output_data, epochs=10, batch_size=32)

'''# Vocabulary size and embedding dimension
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

#Attention mechanism
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
model.fit([padded_input, padded_output], padded_output, batch_size=32, epochs=10)

path = r'C:\BallPredict\[2002-11-01]-0020200026-TOR@SAS.csv'
with open(path):
    model.predict(parseGame(path)[0])'''