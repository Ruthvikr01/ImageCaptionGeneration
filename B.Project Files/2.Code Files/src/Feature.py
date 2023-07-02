from Package import *
from Model import model
all_captions = []
mapping = {}
features = {}
if __name__ == "__main__":
   
    directory = os.path.join(os.getcwd(), '/Users/ardhikiran/Documents/ICG_AI/Datasets/Images')
    for img_name in tqdm(os.listdir(directory)):
        img_path = directory + '/' + img_name
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature

    WORKING_DIR = '/Users/ardhikiran/Documents/ICG_AI/src'
    pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))

    with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)

    with open(os.path.join(os.getcwd(), '/Users/ardhikiran/Documents/ICG_AI/Datasets/captions.txt'), 'r') as f:
        next(f)
        captions_doc = f.read()

   
    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
        len(mapping)

    def clean(mapping):
        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                caption = caption.lower()
                caption = caption.replace('[^A-Za-z]', '')
                caption = caption.replace('\s+', ' ')
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
                captions[i] = caption

    clean(mapping)

    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)

    len(all_captions)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
