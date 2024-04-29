#!/usr/bin/python

# generate ontonotes dataset
# conll files output to ../raw/on_gold_conll/dep
# const files output to ../raw/ontonotes/const
python generate_ontonotes.py

# set up training and test data for the NP classifier
# output to ../data
python build_data.py --dataset 'arrau|ontonotes'

# train the NP classifier and predict the results
# predictions outputed to ../preds/onto_preds.json
python xgboost_classifier.py

# build up the conll-2012 format for singletons
# output to ../data/ontonotes_sg
python build_conll.py

# copy the predicted file to coref models
cp -r ../data/ontonotes_sg ../../coref/data/raw/
