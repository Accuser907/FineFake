# FineFake
This is the dataset for **FineFake : A Multi-domain Knowledge-enhanced Large-scale Dataset for Fake News Detection**

# Getting Started
Follow the instructions to download the dataset. You can download text data, metadata, image data and knowledge data.
The dataset is divided into eight domains: business, culture, daily_life, entertainment, health, politics, sports, technology.
Each domain corresponds to a arrow document named by it's affliated domain. The dataset can be downloaded [here](https://drive.google.com/drive/folders/1Pi_wOzDAGXsqJhdWTr0hCarLXmz-oBio?usp=drive_link).

## Pyarrow file
The data is stored as pyarrow file, it can be opened to dataframe by following codes.
```c
pip install pyarrow
import pyarrow as pa
dataframe = pa.ipc.open_file(file_path).read_pandas()
```
There are 10 columns in arrow file, each attribute and its corresponding meaning is shown in the table below.
| text | image | entity_id | entity | truth | position | description | image_description | image_wiki_id | image_description | image_entity | title |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| news body text | image data(binary format) | text-entity wiki id | text-entity name | label | text-entity position | text-entity description | image-entity description | image-entity wiki id | image-entity description | image-entity name | news title |
