# FineFake
This is the dataset for **FineFake : A Knowledge-Enriched Dataset for Fine-Grained Multi-Domain Fake News Detection**

# Getting Started
Follow the instructions to download the dataset. You can download text data, metadata, image data and knowledge data.
The dataset is divided into six topics and eight platforms: Politics, Entertainment, Business, Health, Society, Conflict. Snopes, Twitter, Reddit, CNN, Apnews, Cdc.gov, Nytimes, Washingtonpost
Each domain corresponds to a pickle document named by it's affliated domain. The dataset can be downloaded [here](https://drive.google.com/drive/folders/1Pi_wOzDAGXsqJhdWTr0hCarLXmz-oBio?usp=drive_link).

## Pyarrow file
The data is stored as pickle file, it can be opened to dataframe by following codes.
```c
pip install pickle
pip install pandas
import pickle as pkl
import pandas as pd
with open(file_name,"rb") as f:
  data_df = pkl.load(f) # data_df is in dataframe 
```
There are 10 columns in pickle file, each attribute and its corresponding meaning is shown in the table below.
| text | image | entity_id | entity | truth | position | description | image_description | image_wiki_id | image_description | image_entity | title |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| news body text | image data(binary format) | text-entity wiki id | text-entity name | label | text-entity position | text-entity description | image-entity description | image-entity wiki id | image-entity description | image-entity name | news title |
