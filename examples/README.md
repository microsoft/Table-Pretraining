# Examples of Tasks

In this directory we provide scripts for different tasks for reproducing our experimental results and fine-tuning our models on your custom datasets more easily.

## 💬 [Table Question Answering](tableqa)

![Example](https://table-pretraining.github.io/assets/tableqa_task.png)

The task of Table Question Answering (TableQA) is to empower machines to answer users' questions over a given table. The resulting answer(s) can be a region in the table, or a number calculated by applying aggregation operators to a specific region.

We regard TableQA as a machine translation task and employ TAPEX to autoregressively output the answer(s).
Therefore, firstly we should convert the original datasets into a compatible format for the backend learning framework (fairseq or HuggingFace). 

Now we support the **one-stop services** for the following datasets, and you can simply run the linked script to accomplish the dataset preparation.
- [WikiSQL (Zhong et al., 2017)](tableqa/process_wikisql_data.py)
- [SQA (Iyyer et al., 2017)](tableqa/process_sqa_data.py)
- [WikiTableQuestions (Pasupat and Liang, 2015)](tableqa/process_wtq_data.py)
> The one-stop service includes the procedure of downloading datasets and pretrained tapex models, truncating long inputs, converting to the fairseq machine translation format, applying BPE tokenization and preprocessing for fairseq model training.

Once the dataset is prepared, you can run the `tableqa/run_model.py` script to train your TableQA models on different datasets.

### Train

### Evaluate

### Predict

## 🔎 Table Fact Verification (Released by Sep. 5)

![Example](https://table-pretraining.github.io/assets/tableft_task.png)

The preprocessing script of table fact verification is a little complicated, and we're still refactoring the code. Please stay tuned!
