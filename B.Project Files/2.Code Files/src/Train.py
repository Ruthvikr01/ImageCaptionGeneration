from Package import *
from Feature import vocab_size,tokenizer,mapping,features,all_captions
from Model import model
max_length = max(len(caption.split()) for caption in all_captions)
if __name__ == "__main__":
    image_ids = list(features.keys())
    split = int(len(image_ids) * 0.90)
    train = image_ids[:split]
    test = image_ids[split:]

    # image feature layers
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    WORKING_DIR = '/Users/ardhikiran/Documents/ICG_AI/src'
    epochs = 25
    batch_size = 32
    steps = len(train) // batch_size

    for i in range(epochs):
        generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    model.save(WORKING_DIR + '/proj_model.h5')
    pass


