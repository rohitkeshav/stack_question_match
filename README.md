# StackOverflow Similar Question recommendation-system clone
The aim of this project was to replicate StackOverflow’s similar questions section, also play-around/get acquainted with NLP and ML algorithms
Base assumption: All titles are questions

Authors - Anandu Anilkumar, Rohit Keshav

## How?
There are 2 core components to this API
   * Cosine Similarity module
   * Classifier

Implemented machine learning classifier along with a similarity algorithm to classify a user’s question 
with the aim to replicate StackOverflow’s similar questions section results

## QuickStart
   1. ```git clone https://github.com/rohitkeshav/stack_question_match.git```
   2. ```cd stack_question_match```
   3. ```pip install requirements```
   4. ```python consume_api.py <query>```

###### Example Usage
    python consume_api.py "what are abstract classes in Java?"

###### Ongoing
   * Building a test module under classification.py, that would test out different machine classifier algorithms 
   * Improving accuracy of the classifier
