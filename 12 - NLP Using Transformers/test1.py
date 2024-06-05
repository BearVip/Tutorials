from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, trainer_utils, AutoModelForMaskedLM,\
    DataCollatorForLanguageModeling, AutoModelForSequenceClassification, pipeline
from transformers_interpret import SequenceClassificationExplainer
import torch
import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import plotly.express as px
from wordcloud import WordCloud

from tutorial_utils import extract_sequence_encoding, get_xy, dummy_classifier, logistic_regression_classifier, evaluate_classifier
import os

df = pd.read_parquet("12 - NLP Using Transformers/NHTSA_NMVCCS_extract.parquet.gzip")
print(f"shape of DataFrame: {df.shape}")
print(*list(zip(df.columns, df.dtypes)), sep="\n")

font_family = "Times New Roman"
font_size = 12

fig = px.bar(df["NUMTOTV"].value_counts().sort_index(), width=640)
fig.update_layout(
    title="number of cases by number of vehicles",
    xaxis_title="number of vehicles",
    yaxis_title="number of cases",
    font=dict(
        family=font_family,
        size=font_size
    )
)
fig.show(config={"toImageButtonOptions": {"format": 'svg', "filename": "num_vehicles"}})
fig.write_image("12 - NLP Using Transformers/figs/num_vehicles.svg")
print("1")

import cairosvg

# Convert SVG to PDF
cairosvg.svg2pdf(url='12 - NLP Using Transformers/figs/num_vehicles.svg', write_to='12 - NLP Using Transformers/figs/num_vehicles.pdf')
# Delete the original SVG file
os.remove("12 - NLP Using Transformers/figs/num_vehicles.svg")