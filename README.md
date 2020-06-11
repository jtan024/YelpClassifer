# YelpClassifer
A Tool to perform topic identification on Yelp reviews using latent Dirichlet allocation (LDA). (please allow LDA files to run for a few minutes as the LDA algorithm takes a while to execute)

## Files:

LDA.ipynb is the file with the LDA model that generates the main topics from the 100,000 Yelp reviews and labels every review. Please install Jupyter Notebook or Jupyter Lab to open and run the ipynb extension files.

OR

LDA.py (within the "Py files" folder) is a converted file which may also be ran on your local terminal with the following command:
**ipython LDA.py**

## Dependencies:
1. numpy
2. pandas
3. re
4. nltk
5. spacy
6. gensim
7. sklearn
8. pyLDAvis
9. matplotlib
10. seaborn 

(example of installation of package: **pip install numpy**)

**Run the following in terminal or command prompt**

python3 -m spacy download en
