# ling-227-final-project

Authorship ID Project
Dov Greenwood, Shalaka Kulkarni, Keaton Mueller

## Description

This project used machine learning and other techniques to create an ensemble model to identify the author of a particular text. The three individual models and the ensemble model achieved high accuracy on test sets using k-fold cross validation (>97%).

The research goal of the project was to determine whether different authors' translations of the same text bear more similarity to one another, or to the individual author's other works. This hypothesis was evaluated by choosing three authors who translated Homer's "Iliad" at various historical periods, and then generating a "Homer" author profile by training the ensemble model on two translations of Homer. The third translation was then to be identified as either being authored by "Homer" or the last author. The results showed that in 2/3 cases, the model attributed the text more to "Homer" than to the actual translator.

## Instructions

Please see `Project.ipynb` for an example of how to use our code to generate results on training and test data. In particular, the `test_model` function defined there will be the most useful in running different configurations and tests.

## Dependencies

- numpy
- pandas
- nltk
- sklearn
- matplotlib
- scipy
