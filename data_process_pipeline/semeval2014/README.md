# Data Processing Pipeline: SemEval 2014
Process:<br>
1. Convert the raw xml file from SemEval 2014 to tsv file format.<br>
2. Clean and process the text data. <br>
3. Get vocabularies of text and aspects<br>
4. Get word vectors from Google News Word2Vec model and store it in a separate file.<br>
5. Convert words in a sentence to index for feeding into the model.<br>

#### To run the above steps:

```$ python run.py```