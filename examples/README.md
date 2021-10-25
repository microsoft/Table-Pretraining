# Examples of Tasks

In this directory we provide scripts for different tasks for reproducing our experimental results and fine-tuning our models on your custom datasets more easily.

## 💬 [Table Question Answering](tableqa)

![Example](https://table-pretraining.github.io/assets/tableqa_task.png)

The task of Table Question Answering (TableQA) is to empower machines to answer users' questions over a given table. The resulting answer(s) can be a region in the table, or a number calculated by applying aggregation operators to a specific region.

In the following, we provide a step-by-step guide for training, evaluating and interacting with our models, whose backend is supported by the awesome [fairseq](https://github.com/pytorch/fairseq) library. 

> It is worth noting that NOW we only wrap and support fairseq style model training / evaluating / interacting, and we will support HuggingFace style model playing in the near future.

### 🍲 Dataset

In this project, we regard TableQA as a machine translation task and employ TAPEX to autoregressively output the answer(s).
Therefore, firstly we should convert the original datasets into a compatible format for the backend learning framework (fairseq or HuggingFace).

Now we support the **one-stop services** for the following datasets, and you can simply run the linked script to accomplish the dataset preparation.
- [WikiSQL (Zhong et al., 2017)](tableqa/process_wikisql_data.py)
- [SQA (Iyyer et al., 2017)](tableqa/process_sqa_data.py)
- [WikiTableQuestions (Pasupat and Liang, 2015)](tableqa/process_wtq_data.py)

Note that the one-stop service includes the procedure of downloading datasets and pretrained tapex models, truncating long inputs, converting to the fairseq machine translation format, applying BPE tokenization and preprocessing for fairseq model training.

By default, these scripts will process data using the dictionary of `tapex.base`. If you want to switch pre-trained models, please change the variable `MODLE_NAME` at line 21.

After one dataset is prepared, you can run the `tableqa/run_model.py` script to train your TableQA models on different datasets.

### 🍳 Train

To train a model, you could simply run the following command, where:
- `<dataset_dir>` refers to directory which contains a `bin` folder such as `dataset/wikisql/tapex.base`
- `<model_path>` refers to a pre-trained model path such as `tapex.base/model.pt`
- `<model_arch>` is a pre-defined model architecture in fairseq such as `bart_base`.

**HINT**: 
- for `tapex.base` or `tapex.large`, `<model_arch>` should be `bart_base` or `bart_large` respectively.
- we would like to raise the readers' attention on the fact that the `accuracy` metric during training indicates the token-level accuracy defined in fairseq instead of the following denotation accuracy. Therefore, the `checkpoint_best.pt` is not always the best one for denotation accuracy. We recommend readers to evaluate all checkpoints using the following command to determine the best one.


```shell
$ python run_model.py train --dataset-dir <dataset_dir> --model-path <model_path> --model-arch <model_arch>
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

### 🍪 Evaluate

Once the model is fine-tuned, we can evaluate it by running the following command, where:
- `<dataset_dir>` refers to directory which contains a `bin` folder such as `dataset/wikisql/tapex.base`
- `<model_path>` refers to a fine-tuned model path such as `checkpoints/checkpoint_best.pt`
- `<sub_dir>` refers to `valid` or `test` for the validation set and test set.
- `<predict_dir>` is used to save the evaluating result, which indicates the correctness of each sample such as `predict_wikisql`

```shell
$ python run_model.py eval --dataset-dir <dataset_dir> --model-path <model_path> --sub-dir <sub_dir> --predict-dir <predict_dir>
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

### 🍻 Interact

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
$ python run_model.py predict --resource-dir <resource_dir> --checkpoint-name <model_name>
```
> Note that if <resource_dir> is under the current working directory, you should still specify a prefix `./` to make the path like a local path (e.g., ./tapex.base). Otherwise, fairseq will regard it as a model name.

## 🔎 Table Fact Verification

![Example](https://table-pretraining.github.io/assets/tableft_task.png)

The task of Table Fact Verification (TableFT) is to empower machines to justify if a statement follows facts in a given table. The result is a binary classification belonging to `1` (yes) or `0` (no).

### 🍲 Dataset

In this project, following the practise of BART on sequence classification tasks, we feed the same input to the encoder and the decoder of TAPEX, and build a binary classifier on top of the hidden state of the last token in the decoder to output `0` or `1`.
Similar to the one in TableQA, the first step is to convert the original dataset into a compatiable format with fairseq.

Now we support the **one-stop services** for the following datasets, and you can simply run the linked script to accomplish the dataset preparation.
- [TabFact (Chen et al., 2020)](tableft/process_tabfact_data.py)

Note that the one-stop service includes the procedure of downloading datasets and pretrained tapex models, truncating long inputs, converting to the fairseq sentence classification format, applying BPE tokenization and preprocessing for fairseq model training.

By default, these scripts will process data using the dictionary of `tapex.base`. If you want to switch pre-trained models, please change the variable `MODLE_NAME` at line 21.

After one dataset is prepared, you can run the `tableft/run_model.py` script to train your TableFT models.

### 🍳 Train

To train a model, you could simply run the following command, where:
- `<dataset_dir>` refers to directory which contains a `input0` and a `label` folder such as `dataset/tabfact/tapex.base`
- `<model_path>` refers to a pre-trained model path such as `tapex.base/model.pt`
- `<model_arch>` is a pre-defined model architecture in fairseq such as `bart_base`.

**HINT**:
- for `tapex.base` or `tapex.large`, `<model_arch>` should be `bart_base` or `bart_large` respectively.
- the reported `accuracy` metric during training is the offcial binary classification accuracy defined in TabFact.

```shell
$ python run_model.py train --dataset-dir <dataset_dir> --model-path <model_path> --model-arch <model_arch>
```

A full list of training arguments can be seen as below:

```
--dataset-dir DATASET_DIR
                    dataset directory where train.src is located in
--exp-dir EXP_DIR   
                    experiment directory which stores the checkpoint
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

### 🍪 Evaluate

Once the model is fine-tuned, we can evaluate it by running the following command, where:
- `<dataset_dir>` refers to directory which contains a `.input0` and a `.label` file such as `dataset/tabfact`. **ATTENTION, THIS IS NOT THE SAME AS IN TABLEQA**.
- `<model_dir>` refers to directory which contains a fine-tuned model as `model.pt` such as `checkpoints`.
- `<sub_dir>` refers to `valid`, `test`, `test_simple`, `test_complex`, `test_small` for different testing.

```shell
$ python run_model.py eval --dataset-dir <dataset_dir> --model-dir <model_dir> --sub-dir <sub_dir>
```

A full list of evaluating arguments can be seen as below:

```
--dataset-dir DATASET_DIR
                    dataset directory where train.src is located in
--model-dir MODEL_DIR
                    the directory of fine-tuned model path such as
                    wikisql.tapex.base
--sub-dir {train,valid,test,test_complex,test_simple,test_small}
                    the directory of pre-trained model path, and the
                    default should be in{bart.base, bart.large,
                    tapex.base, tapex.large}.
```

## 🏋🏻 [Table Pre-training](pretrain)

The procedure is as introduced in TableQA, and please follow the same procedure with scripts under [pretrain](pretrain) to perform pre-training on the pre-training corpus!
If you'd like to pre-train the model with your data (e.g., private data), you should prepare them as the same format as the released table pre-training corpus, which is as following:
```shell
- train.src # inputs for training, one line one input
- train.tgt # outputs for training, one line one output
- valid.src (optional) # inputs for validation, one line one input
- valid.tgt (optional) # outputs for validation, one line one output
```

> If `valid.src` and `valid.tgt` are not provided, the script will automatically take a random set of `20,000` examples from the training set as the validation set.

Also, if you would like to probe the SQL execution performance, the `predict` mode in [run_model.py](pretrain/run_model.py) would be your best choice.
As done in above TableQA, you can pass an SQL query and a Table into TAPEX, and it returns its **execution** result.