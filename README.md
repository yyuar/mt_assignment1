# German-English translator
This project aims to train a German-English translator based on the attentional neural model.
The dataset for training is IWSLT German-English dataset.

## Algorithm Design
Basically follow the framework given in the pseudocode, which can be divided into training and translation.
Both parts include:
  - Encoding: Using mini batch in training process, do the padding and deal with the masks. 
  - Decoding: Feed in an endsymbol and extract an initial state. Using this state and the encoding of source sentence to compute the first context in order to get the prediction. Then compute the loss for the first word.

## How to run:
1. Start a p2.xlarge instance on aws.
2. Use AMI provided by course instructor.
3. GPU+ at least 11G memory
4. run "python MT_mini.py"

## Estimated Time
Each epoch takes about 30 minutes.
