This project implements a multitask learning model for BERT models. This can perform three downstream tasks: Sentiment Analysis, Sarcasm Detection and negation detection between a pair of sentences.
# **Running the code**
The model can be run by ``` python main.py individual ``` or ``` python main.py multitask ```command. 

First one will train three models on three tasks independently. Second one will train one single multi task model.

The performances will be printed. The logs and results can be visualized in the respective folders.

It can be run with ``` --transformer bert ``` or ``` --transformer roberta ``` to use the particular model as base model.

For more details on option, refer the ``` main.py``` file.
