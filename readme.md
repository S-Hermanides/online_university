## Readme

### Introduction

Welcome to the code for my project about improving student Pass / Fail ratio in an online university. This was the second project I worked on during my Metis Data Science Bootcamp.

I used a dataset from the Open University, specifically their open dataset which can be found [here](https://analyse.kmi.open.ac.uk). I re-created the tables in Postgresql on an AWS machine and used SQL queries to pull the relevant information from there. 

In the code folder for this project you'll find the following files:

* Data_extract_clean
  * This notebook extracts the data from the Postgresql database and cleans it into a central table for the classification problem
* Eda_modeling
  * This notebook goes through Exploratory Data Analysis (EDA) and model selection
* model_select.py
  * Helper functions for the modeling notebook, specifically for the cross-validation pipeline and a grid search of parameters
