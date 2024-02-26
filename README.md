# FineFake
This is the dataset for **FineFake : A Knowledge-Enriched Dataset for Fine-Grained Multi-Domain Fake News Detection**

# Getting Started
Follow the instructions to download the dataset. You can download text data, metadata, image data and knowledge data.
The dataset is divided into six topics and eight platforms: Politics, Entertainment, Business, Health, Society, Conflict. Snopes, Twitter, Reddit, CNN, Apnews, Cdc.gov, Nytimes, Washingtonpos. The dataset and images can be downloaded [here](https://drive.google.com/file/d/1IwkI1Ppr24ICebKMqUY51csqOim56LkD/view?usp=sharing).

## DataFrame file
The data is stored as pickle file, it can be opened to dataframe by following codes.
```c
pip install pickle
pip install pandas
import pickle as pkl
import pandas as pd
with open(file_name,"rb") as f:
  data_df = pkl.load(f) # data_df is in dataframe 
```
There are 13 columns in pickle file, each attribute and its corresponding meaning is shown in the table below.
| text | image_path | entity_id | topic | label | fine-grained label | knowledge_embedding | description | relation | platform | author | date | comment |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| news body text | image_path(relative path) | text-entity wiki id | topic from six topics | label | fine-grained label | knowledge_embedding | text-entity description | relation | The source of the news | author | The date of the news publication | comment |
