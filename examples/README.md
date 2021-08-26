# Examples of Tasks

In this directory we provide scripts for different tasks for reproducing our experimental results and fine-tuning our models on your custom datasets more easily.

## 💬 [Table Question Answering](tableqa)

![Example](https://table-pretraining.github.io/assets/tableqa_task.png)

The task of Table Question Answering (TableQA) is to empower machines to answer users' questions over a given table. The resulting answer(s) can be a region in the table, or a number calculated by applying aggregation operators to a specific region.

In the following, we provide a step-by-step guide for training, evaluating and interacting with our models, whose backend is supported by the awesome [fairseq](https://github.com/pytorch/fairseq) library. 

> It is worth noting that NOW we only wrap and support fairseq style model training / evaluating / interacting, and we will support HuggingFace style model playing in the near future.

### Dataset

In this project, we regard TableQA as a machine translation task and employ TAPEX to autoregressively output the answer(s).
Therefore, firstly we should convert the original datasets into a compatible format for the backend learning framework (fairseq or HuggingFace).

Now we support the **one-stop services** for the following datasets, and you can simply run the linked script to accomplish the dataset preparation.
- [WikiSQL (Zhong et al., 2017)](tableqa/process_wikisql_data.py)
- [SQA (Iyyer et al., 2017)](tableqa/process_sqa_data.py)
- [WikiTableQuestions (Pasupat and Liang, 2015)](tableqa/process_wtq_data.py)

Note that the one-stop service includes the procedure of downloading datasets and pretrained tapex models, truncating long inputs, converting to the fairseq machine translation format, applying BPE tokenization and preprocessing for fairseq model training.

After one dataset is prepared, you can run the `tableqa/run_model.py` script to train your TableQA models on different datasets.

### Train Model

To train a model, you could simply run the following command, where `<dataset_dir>` refers to dirs such as `dataset/wikisql`, and `<model_path>` refers to a pre-trained model path such as `bart.base/model.pt`.

```shell
python run_model.py train --dataset-dir <dataset_dir> --model-path <model_path>
```

A full list of training arguments can be seen as below:

```
--dataset-dir DATASET_DIR
                    dataset directory where train.src is located in
--exp-dir EXP_DIR     experiment directory which stores the checkpoint
                    weights
--model-path MODEL_PATH
                    the directory of pre-trained model path
--model-arch {bart_large,bart_base}
                    tapex large should correspond to bart_large, and tapex base should be bart_base
--max-tokens MAX_TOKENS
                    if you train a large model on 16GB memory, max-tokens
                    should be empirically set as 1536, and can be near-
                    linearly increased according to your GPU memory.
--gradient-accumulation GRADIENT_ACCUMULATION
                    the accumulation steps to arrive a equal batch size,
                    the default value can be usedto reproduce our results.
                    And you can also reduce it to a proper value for you.
--total-num-update TOTAL_NUM_UPDATE
                    the total optimization training steps
--learning-rate LEARNING_RATE
                    the peak learning rate for model training
```

### Evaluate Model

Once the model is fine-tuned, we can evaluate it by runing the following command, where `<dataset_dir>` refers to dirs such as `dataset/wikisql`, and `<model_path>` refers to a fine-tuned model path such as `checkpoints/checkpoint_best.pt`.

```shell
python run_model.py eval --dataset-dir <dataset_dir> --model-path <model_path>
```

A full list of evaluating arguments can be seen as below:

```
--dataset-dir DATASET_DIR
                    dataset directory where train.src is located in
--model-path MODEL_PATH
                    the directory of fine-tuned model path such as
                    wikisql.tapex.base
--sub-dir {train,valid,test}
                    the directory of pre-trained model path, and the
                    default should be in{bart.base, bart.large,
                    tapex.base, tapex.large}.
--max-tokens MAX_TOKENS
                    the max tokens can be larger than training when in
                    inference.
--predict-dir PREDICT_DIR
                    the predict folder of generated result.
```

### Interact with Model

Except for offline generating, we also wrap a model interface for interacting with our model, which will be useful for online prediction.

Different from evaluating, the interacting requires developers to put the best model checkpoint along with its corresponding resources into the same directory.
What is a resource directory? It is the folder which stores the following files:
- encoder.json
- vocab.bpe
- dict.txt
- model.pt (*the best model checkpoint*)

You can find it in downloaded resource folders `bart.base`, `bart.large`, `tapex.base`, `tapex.large` when preparing datasets.

Then you can predict the answer online with the following command, where `<model_name>` refers to the model weight file name such as `model.pt`.

```shell
python run_model.py predict --resource-dir <resource_dir> --checkpoint-name <model_name>
```

## 🔎 Table Fact Verification (Released by Sep. 5)

![Example](https://table-pretraining.github.io/assets/tableft_task.png)

The preprocessing script of table fact verification is a little complicated, and we're still refactoring the code. Please stay tuned!
