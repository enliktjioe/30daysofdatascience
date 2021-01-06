#!/usr/bin/env python
# coding: utf-8

# ## References
# 
# - https://apple.github.io/turicreate/docs/userguide/
# - https://apple.github.io/turicreate/docs/userguide/recommender/
# - https://apple.github.io/turicreate/docs/userguide/sframe/sframe-intro.html
# - https://apple.github.io/turicreate/docs/userguide/vis/gallery.html
# - https://apple.github.io/turicreate/docs/userguide/image_classifier/

import turicreate as tc
import pandas as pd
import numpy as np #test


# ## Read

get_ipython().run_cell_magic('time', '', "df = tc.SFrame.read_csv('data/data_1gb.csv')\ndf.head()")


get_ipython().run_cell_magic('time', '', "df_pd = pd.read_csv('data/data_1gb.csv')\ndf_pd.head()")


# ## Write

get_ipython().run_cell_magic('time', '', "df.export_csv('output/sframe_1gb.csv')")


get_ipython().run_cell_magic('time', '', "df_pd.to_csv('output/df_pd_1gb.csv')")


# ## Recommender System

import turicreate as tc
actions = tc.SFrame.read_csv('data/ml-20m/ratings.csv')


items = tc.SFrame.read_csv('data/ml-20m/movies.csv')


## creating a model
training_data, validation_data = tc.recommender.util.random_split_by_user(actions, 'userId', 'movieId')
model = tc.recommender.create(training_data, 'userId', 'movieId')


results = model.recommend()


data = tc.SFrame({'user_id': ["Ann", "Ann", "Ann", "Brian", "Brian", "Brian"],
                          'item_id': ["Item1", "Item2", "Item4", "Item2", "Item3", "Item5"],
                          'rating': [1, 3, 2, 5, 4, 2]})
m = tc.factorization_recommender.create(data, target='rating')

recommendations = m.recommend()


recommendations = m.recommend(users=['Brian'])
recommendations


# ## Text Classifier

import turicreate as tc

# Only load the first two columns from csv
data = tc.SFrame.read_csv('data/classifier = model.classifierSMSSpamCollection', header=False, delimiter='\t', quote_char='\0')

# Rename the columns
data = data.rename({'X1': 'label', 'X2': 'text'})

# Split the data into training and testing
training_data, test_data = data.random_split(0.8)

# Create a model using higher max_iterations than default
model = tc.text_classifier.create(training_data, 'label', features=['text'], max_iterations=100)

# Save predictions to an SArray
predictions = model.predict(test_data)

# Make evaluation the model
metrics = model.evaluate(test_data)
print(metrics['accuracy'])

# Save the model for later use in Turi Create
model.save('MyTextMessageClassifier.model')

# Export for use in Core ML
model.export_coreml('MyTextMessageClassifier.mlmodel')


classifier = model.classifier


model.export_coreml('MyTextMessageClassifier.mlmodel')


# ## Visualization

import turicreate as tc

df_viz = tc.SFrame.read_csv('data/cell.csv')
df_viz.show()


df_viz.explore()


sf = tc.SFrame.read_csv('https://docs-assets.developer.apple.com/turicreate/datasets/tc-clang-format-results.csv')
sf.explore()


# ## Image Classification

import turicreate as tc

# Load images (Note:'Not a JPEG file' errors are warnings, meaning those files will be skipped)
data = tc.image_analysis.load_images('data/PetImages', with_path=True)

# From the path-name, create a label column
data['label'] = data['path'].apply(lambda path: 'dog' if '/Dog' in path else 'cat')

# Save the data for future use
data.save('cats-dogs.sframe')

# Explore interactively
data.explore()


import turicreate as tc

# Load the data
data =  tc.SFrame('cats-dogs.sframe')

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Create the model
model = tc.image_classifier.create(train_data, target='label')

# Save predictions to an SArray
predictions = model.predict(test_data)

# Evaluate the model and print the results
metrics = model.evaluate(test_data)
print(metrics['accuracy'])

# Save the model for later use in Turi Create
model.save('cats-dogs.model')

# Export for use in Core ML
model.export_coreml('MyCustomImageClassifier.mlmodel')


new_cats_dogs = tc.image_analysis.load_images('data/PetImages', with_path=True)


new_cats_dogs['predictions'] = model.predict(new_cats_dogs)


new_cats_dogs['predictions']


new_cats_dogs.head().explore()


new_cats_dogs.tail().explore()




