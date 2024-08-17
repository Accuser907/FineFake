# FineFake
This is the dataset for **FineFake : A Knowledge-Enriched Dataset for Fine-Grained Multi-Domain Fake News Detection**. The paper can be downloaded [here](https://doi.org/10.48550/arXiv.2404.01336). The main construction of FineFake is shown below. The code of construction for updating latest news will be released when the paper is accepted.
![construction_00](https://github.com/Accuser907/FineFake/assets/61140633/dbf1af33-9cc8-4f1d-9208-6be46a88fe54)

# Getting Started
Follow the instructions to download the dataset. You can download text data, metadata, image data and knowledge data.
The dataset is divided into six topics and eight platforms: Politics, Entertainment, Business, Health, Society, Conflict. Snopes, Twitter, Reddit, CNN, Apnews, Cdc.gov, Nytimes, Washingtonpos. The dataset and images can be downloaded [here](https://drive.google.com/file/d/1IwkI1Ppr24ICebKMqUY51csqOim56LkD/view?usp=sharing).

## DataFrame file
The data is stored as pickle file, it can be opened to dataframe by following codes. Details can be found at demo.py.
```c
pip install pickle
pip install pandas
import pickle as pkl
import pandas as pd
with open(file_name,"rb") as f:
  data_df = pkl.load(f) # data_df is in dataframe 
```
There are 13 columns in pickle file, each attribute and its corresponding meaning is shown in the table below:
| text | image_path | entity_id | topic | label | fine-grained label | knowledge_embedding | description | relation | platform | author | date | comment |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| news body text | image_path(relative path) | text-entity wiki id | topic from six topics | label | fine-grained label | knowledge_embedding | text-entity description | relation | The source of the news | author | The date of the news publication | comment |

## Labels
For the binary label, "0" represents fake and "1" represents real.
For the fine-grained label, each label and its corresponding meaning is shown in the table below:
| 0 | 1 | 2 | 3 | 4 | 5 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| real | text-image inconsistency | content-knowledge inconsistency | text-based fake | image-based fake | others |

## Guidelines
- FineFake is designed to advance research in fake news detection and should not be used for any malicious or harmful purposes.  Users should refrain from using the dataset for generating or spreading misinformation, manipulating public opinion, or any other activity that could harm individuals, groups, or society at large.
- It is the responsibility of users to ensure that their models and research outcomes are fair and unbiased. Any biases inherent in the dataset must be carefully addressed in your work. If biases are detected, they should be documented, and appropriate mitigation strategies should be applied.
- The FineFake dataset contains data sourced from public domains, but it is essential to respect the privacy and anonymity of individuals. Any attempt to de-anonymize individuals or re-identify entities within the dataset is strictly prohibited. All users must ensure that their research upholds the principles of privacy protection.
