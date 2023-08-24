import tensorflow as tf
import numpy as np
import json

# Load data from JSON file


def load_data_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    input_texts = []
    output_texts = []
    for item in data:
        input_text = item['input']
        output_text = item['output']
        input_texts.append(input_text)
        output_texts.append(output_text)
    return input_texts, output_texts

# Tokenize input and output sequences


def tokenize_sequences(input_texts, output_texts):
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    input_tokenizer.fit_on_texts(input_texts)
    input_sequences = input_tokenizer.texts_to_sequences(input_texts)
    max_input_length = max(len(seq) for seq in input_sequences)
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        input_sequences, maxlen=max_input_length)

    output_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    output_tokenizer.fit_on_texts(output_texts)
    output_sequences = output_tokenizer.texts_to_sequences(output_texts)
    max_output_length = max(len(seq) for seq in output_sequences)
    output_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        output_sequences, maxlen=max_output_length)

    return input_sequences, output_sequences, input_tokenizer, output_tokenizer

# Create the Seq2Seq model


def create_model(input_vocab_size, output_vocab_size, embedding_dim, hidden_units):
    encoder_inputs = tf.keras.layers.Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(
        input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(hidden_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(
        output_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(
        hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(
        output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs)
    return model


print(0)
# Load data from JSON file
input_texts, output_texts = load_data_from_json('mariya_anti_data.json')

# Tokenize input and output sequences
print(1)
input_sequences, output_sequences, input_tokenizer, output_tokenizer = tokenize_sequences(
    input_texts, output_texts)

# Define model parameters
embedding_dim = 128
hidden_units = 256

print(2)
# Create the Seq2Seq model
model = create_model(len(input_tokenizer.word_index) + 1,
                     len(output_tokenizer.word_index) + 1, embedding_dim, hidden_units)

print(3)
# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')
model.fit([input_sequences, output_sequences[:, :-1]],
          np.expand_dims(output_sequences[:, 1:], -1), batch_size=1, epochs=50)

print(4)
# Save the model
model.save('seq2seq_model.h5')

print(5)
# Generate transformed output using the trained model


print(6)


def generate_transformed_output(input_text):
    input_sequence = input_tokenizer.texts_to_sequences([input_text])
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        input_sequence, maxlen=input_sequences.shape[1])
    output_sequence = np.zeros((1, output_sequences.shape[1]))

    for i in range(output_sequences.shape[1]):
        predictions = model.predict(
            [input_sequence, output_sequence]).argmax(axis=-1)
        output_sequence[0, i] = predictions[0, i]

    output_text = output_tokenizer.sequences_to_texts(output_sequence)[0]
    return output_text


print(7)

# Test the model with a sample input
input_text = "To \nGeetha Balakumar \nNew LIG 62 , \nHousing board sector, \nOpposite to primary health care centre,\nAnnanagar , \nMadurai 625020\n\nPhone number: 9944274478\n\nFrom \nMythrayee \n9840322745.\n\nSize: 2.6"
transformed_output = generate_transformed_output(input_text)
print("Input:", input_text)
print("Transformed Output:", transformed_output)

print(8)
