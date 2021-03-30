# rilla-voice-tech-challenge

Models Trained

1. One hot encoding clustering on cleaned data

        Average silhouette_score : 0.025480204169789522

    ![one_hot_elbow.png](data/results/plots/one_hot_elbow.png)
    ![one_hot_cluster.png](data/results/plots/one_hot_cluster.png)

    Cleaning Pipeline:
    a. Use spacy to detect the entities and replace the entities text as entities. eg. replaced google as org
    b. removed stopwords and lemmatized

    To run the model:

        cd modules/clustering
        python one_hot_clustering.py

2. Tf-Idf Clustering on cleaned data

        Average silhouette_score : 0.006704295826363957

    ![tf_idf_elbow.png](data/results/plots/tf_idf_elbow.png)
    ![tf_idf_cluster.png](data/results/plots/tf_idf_cluster.png)

    Cleaning Pipeline:
    a. Use spacy to detect the entities and replace the entities text as entities. eg. replaced google as org
    b. removed stopwords and lemmatized

    To run the model:
    
        cd modules/clustering
        python tf_idf_clustering.py

3. Fasttext vectors trained on uncleaned data

        Average silhouette_score : 0.17497733

    ![fasttext_uncleaned_elbow.png](data/results/plots/fasttext_uncleaned_elbow.png)
    ![fasttext_uncleaned_cluster.png](data/results/plots/fasttext_uncleaned_cluster.png)

    Cleaning Pipeline:
    No Cleaning

    To run the model:
    
        cd modules/clustering
        set cleaning variable as False
        python fasttext_clustering.py


4. Fasttext vectors trained on cleaned data

        Average silhouette_score : 0.15030155

    ![fasttext_cleaned_elbow.png](data/results/plots/fasttext_cleaned_elbow.png)
    ![fasttext_cleaned_cluster.png](data/results/plots/fasttext_cleaned_cluster.png)

    Cleaning Pipeline:
    a. Use spacy to detect the entities and replace the entities text as entities. eg. replaced google as org
    b. removed stopwords and lemmatized

    To run the model:
    
        cd modules/clustering
        set cleaning variable as True
        python fasttext_clustering.py

5. Sent2vec clustering

        Average silhouette_score : 0.110265754

    sent2vec_
    ![sent2vec_elbow.png](data/results/plots/sent2vec_elbow.png)
    ![sent2vec_cluster.png](data/results/plots/sent2vec_cluster.png)

    Cleaning Pipeline:
    No Cleaning

    To run the model:
    
        cd modules/clustering
        python sent2vec_clustering.py

## Instructions

Add the path to the git folder in the PYTHONPATH

    export PYTHONPATH="$PYTHONPATH:~/rilla-voice-tech-challenge"