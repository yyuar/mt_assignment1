This project aims to train a German-English translator based on the attentional neural model.
The dataset for training is IWSLT German-English dataset.
I basically follow the framework given in the pseudocode, which can be divided into following parts: training(encoding, decoding)(encoding: Since I used mini batch in my training process, I need to do the padding and deal with the masks. decoding:As for the decoding, I first feed in an endsymbol and extract an initial state. I used this state and
the encoding of source sentence to compute the first context in order to further get the prediction
and compute the loss for the first word.) and translation(encoding, decoding).

How to run:
1. start a p2.xlarge instance on aws.
2. Use AMI provided by course instructor.
3. GPU+at least 11G memory
Each epoch takes about 30 minutes.
