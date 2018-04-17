# Artificial Intelligence Engineer Nanodegree

## Probabilistic Models: A Sign Language Recognition System project using Hidden Markov Models

[image1]: ./recognizer_screenshot.png "AIND-Recognizer Screenshot"

### Motivation

The main goal of this project is to build a word recognizer for some American Sign Language video sequences using hidden Markov models using features extracted from gestural measurements taken from videos frames collected for research (see the RWTH-BOSTON-104 Database).

#### PART 1: Extracting the features

A data handler has been designed for this database provided as AslDb class. This module creates the pandas dataframe from the corpus and dictionaries for extracting data [hmmlearn library](https://hmmlearn.readthedocs.io/en/latest/) format-friendly. Every video frame can be expressed with an id and its features. Features are the most relevant variables that will allow us to train the model. They can be raw measurements directly taken from the video frames or derived from them like normalize each speaker's range of motion with grouped statistics using Pandas stats functions and pandas groupby. 	

#### PART 2: Training the HMM Model: Finding the optimal hidden states

Once extracting the features, we need to train the model but the number of optimal hidden states is unknown. The purpose the model selection is to tune the number of states for each word HMM prior to testing on unseen data. Three methods are explored: Log-likelihood using cross-validation folds (CV), Bayesian Information Criterion (BIC), Discriminative Information Criterion (DIC). 

**SelectorCV** uses cross-validation using k-folds, that means that we use a subset of data for training, and another subtest, a testing dataset to validate the model. The testing dataset is not used for training, that means at the moment of model validation, this dataset has remained unseen by the model. This way, we can see how well the model is behaving generalizing data, or if there is overfitting due to an excess of complexity. The disadvantage of this method is the need additional time to preprocess the data -split and randomize the k buckets- since we need different datasets. Besides, we need a dataset big enough to generate k buckets of data. The second drawback is the performance since it uses k buckets, that means that we train the model k-1 times, and leave one bucket for testing. This process is performed k times, thus it might take longer to train and validate the model although, certainly every bucket is shorter compared to the whole dataset. Usual values are k=5 or k=10 (from AIMA book), anyway, more k-folds, more computationally expensive. Besides, a metrics for validation have to be defined [http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter], however, the problem of the metric choice is no dependent on the train-test split. In this case, we use the mean get for every bucket.


**Bayesian Information Criterion** or **BIC** tries to minimize the resulting score. For this implementation, we use BIC = −2 log L + p log N. Both terms are straightly related to model complexity. Therefore we use this method to prevent overfitting since as more complexity added to the model, a higher penalization. The greatest advantage compared to SelectorCV or to DIC, is that we are going to find an optimal solution here, as model complexity is balanced with the accuracy obtained, and that is probably why the performance for this selector, is the best obtained.


**Discriminative Information Criterion** or **DIC** is focused on the classification task itself. The score is obtained by getting the likelihood of a known word, compared to the mean of anti-likelihood, that is the probability of being another word, which is going to be very low. As a result, we have got a number of competing models, however, the chosen model is the one which maximizes the likelihood -the probability of being a word 'whatever', and minimizes the anti-likelihood -the probability that the word 'whatever' is not 'outside', for instance. Due to DIC works with likelihoods as well as anti-likelihoods, it performs better with a large number of states. Besides, it does not care about model complexity, thus it has the lowest performance.

#### PART 3: BUILD THE RECOGNIZER

Finally, we get a dict with all the unique words we are going to use for training, and a GaussianHMM object as value with a probability distribution given the word. This dict is going to be used on unseen data and try to predict sequences and calculate Word error rate (WER), the common metric of the performance of a speech recognition or machine translation system, given different features and selectors criterion chosen.    

![AIND-Recognizer][image1]


### Modules available

* ```asl_data.py``` contains a data handler designed for this database provided as AslDb class. This module creates the pandas dataframe from the corpus and dictionaries for extracting data [hmmlearn library](https://hmmlearn.readthedocs.io/en/latest/) format-friendly. 	Every video frame can be expressed with an id and its features. 
* ```asl_utils.py``` provides unit tests as a sanity check on the defined feature sets. The test simply looks for some valid values.
* ```asl_test_*.py``` provides unit tests for all the different components of this notebook: recognizer, model selectors, etc. 
* ```my_model_selectors.py```contains the SelectorCV, SelectorBIC, and SelectorDIC classes.
* ```my_recognizer.py``` contains the dictionary of words and models to recognize test word sequences from word models set

### Install environment, Test

* [Install instructions](https://github.com/udacity/AIND-Recognizer)
* [Test](http://localhost:8888/notebooks/AIND-Recognizer/asl_recognizer.ipynb)
* [Demo](http://localhost:8888/notebooks/AIND-Recognizer/asl_recognizer.ipynb)

#### Provided Raw Data

The data in the `asl_recognizer/data/` directory was derived from 
the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php). 
The handpositions (`hand_condensed.csv`) are pulled directly from 
the database [boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml](boston104.handpositions.rybach-forster-dreuw-2009-09-25.full.xml). The three markers are:

*   0  speaker's left hand
*   1  speaker's right hand
*   2  speaker's nose
*   X and Y values of the video frame increase left to right and top to bottom.

Take a look at the sample [ASL recognizer video](http://www-i6.informatik.rwth-aachen.de/~dreuw/download/021.avi)
to see how the hand locations are tracked.

The videos are sentences with translations provided in the database.  
For purposes of this project, the sentences have been pre-segmented into words 
based on slow motion examination of the files.  
These segments are provided in the `train_words.csv` and `test_words.csv` files
in the form of start and end frames (inclusive).

The videos in the corpus include recordings from three different ASL speakers.
The mappings for the three speakers to video are included in the `speaker.csv` 
file.
