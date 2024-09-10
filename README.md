## **GUIT-AsTourNE: A Dataset of Assamese Named Entities in the Tourism Domain**

This repository contains GUIT-AsTourNE dataset as well as the best-performing pre-trained model. The dataset is split into train(70%), dev(15%), and test(15%).

### Usage:

In order to run this model, first download the pretrained model GUIT-AsTourNER from [here](https://drive.google.com/file/d/11TpzfV6MsOBVQyhROb__Tis85QcMNdWB/view?usp=sharing). Unzip the downloaded model.

```

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("path_to_model")
model = AutoModelForTokenClassification.from_pretrained("path_to_model")

def get_tag(sentence, tokenizer, model):
    tok_sentence = tokenizer(sentence, return_tensors='pt')

    with torch.no_grad():
        logits = model(**tok_sentence).logits.argmax(-1)

    predicted_tokens_tag = [model.config.id2label[t.item()] for t in logits[0]]

    predicted_labels = []

    previous_token_id = 0
    word_ids = tok_sentence.word_ids()
    for word_index in range(len(word_ids)):
        if word_ids[word_index] == None:
            previous_token_id = word_ids[word_index]
        elif word_ids[word_index] == previous_token_id:
            previous_token_id = word_ids[word_index]
        else:
            predicted_labels.append(predicted_tokens_tag[ word_index ])
            previous_token_id = word_ids[word_index]

    return predicted_labels


sentence = "১৮০৬ চনত স্বৰ্গদেউ কমলেশ্বৰ সিংহ জয়সাগৰৰ মন্দিৰ পৰিদৰ্শন কৰিছিল ।"
prediction = get_tag(sentence=sentence, tokenizer=tokenizer, model=model)

for index in range(len(sentence.split(' '))):
    print(sentence.split(' ')[index] + '\t' + prediction[index])

```
#### Result:
<pre>
১৮০৬    B-YEAR  
চনত    I-YEAR  
স্বৰ্গদেউ    O  
কমলেশ্বৰ    B-PER  
সিংহ    I-PER  
জয়সাগৰৰ    B-LOC  
মন্দিৰ    I-LOC  
পৰিদৰ্শন    O  
কৰিছিল    O  
।    O
</pre>
