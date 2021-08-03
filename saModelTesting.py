from transformers import pipeline
import torch

classifier = pipeline('sentiment-analysis')

classifier = classifier.to(torch.device('cuda'))


results = classifier(["this cookie tastes great",
                      "this cookie tastes like a dried paper towel.",
                      'this cookie tastes like it has been sitting for centuries',
                      'this cookie should be a great whetstone for knives',

                      'you really did a good gob',
                      'you really do a good gob baking these unappetizing cookies'])


for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")





