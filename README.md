# T<span style="font-size:0.8em;">A</span>PE<span style="font-size:0.8em;">X</span>: Table Pre-training via Learning a Neural SQL Executor

The official repository which contains the code and pre-trained models for our paper [T<span style="font-size:0.8em;">A</span>PE<span style="font-size:0.8em;">X</span>: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/pdf/2107.07653.pdf).

# 🔥 Updates

- 2021-08-27: We released the code, the pre-training corpus, and the pre-trained TAPEX model weights. Thanks for your patience!
- 2021-07-16: We released our [paper](https://arxiv.org/pdf/2107.07653.pdf) and [home page](https://table-pretraining.github.io/). Check it out!

# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview

## 📝 Paper

In the paper, we present T<span class="span-small">A</span>PE<span class="span-small">X</span> (for **Ta**ble **P**re-training via **Ex**ecution), a conceptually simple and empirically powerful pre-training approach to empower existing models with table reasoning skills.
T<span class="span-small">A</span>PE<span class="span-small">X</span> realizes table pre-training by **learning a neural SQL executor over a synthetic corpus**, which is obtained by automatically synthesizing executable SQL queries.

<figure style="text-align:center">
  <img src="https://table-pretraining.github.io/assets/tapex_overview.jpg" width="300">
  <figcaption>Fig 1. The schematic illustration of T<span class="span-small">A</span>PE<span class="span-small">X</span>. Tables not shown for brevity.</figcaption>
</figure>


The central point of T<span class="span-small">A</span>PE<span class="span-small">X</span> is to train a model to **mimic the SQL query execution process over a table**.
We believe that if a model can be trained to faithfully *execute* SQL queries, then it must have a deep understanding of table structures and possess an inductive bias towards table structures.


<div style="text-align:center">
<img src="https://table-pretraining.github.io/assets/model_pretrain.gif" width="600"></div>


Meanwhile, since the diversity of SQL queries can be guaranteed systemically, and thus a *diverse* and *high-quality* pre-training corpus can be automatically synthesized for T<span class="span-small">A</span>PE<span class="span-small">X</span>.

## 💻 Project

This project contains two parts, `tapex` library and `examples` to employ it on different table-related applications (e.g., Table Question Answering).

- For `tapex`, there is an overview:

```shell
|-- common
    |-- dbengine.py # the database engine to return answer for a SQL query
    |-- download.py # download helper for automatic resource
|-- data_utils
    |-- wikisql
        |-- executor.py # the re-implementation of WikiSQL style SQL execution to obtain ground-truth answers in the dataset
    |-- format_converter.py # convert dataset formats into HuggingFace style
    |-- preprocess_binary.py # wrapper for the fairseq preprocess script
    |-- preprocess_bpe.py # wrapper for the BPE preprocess
|-- processor
    |-- table_linearize.py # the class to flatten a table into a linearized form, which should keep consistent during pre-training, fine-tuning and evaluating
    |-- table_truncate.py # the class to truncate a long table into a shorter version to satisfy model's input length limit (e.g., BART can accept at most 1024 tokens)
    |-- table_processor.py # the wrapper for the above two table utility function classes
|-- model_eval.py # evaluate the denotation accuracy of model
|-- model_interface.py # wrap a model interface for interaction based on HubInterface
```

- For `examples`, please refer to [here](examples) for more details.

# ⚡️ Quickstart

## Prepare Environment

First, you should set up a python environment. This code base has been tested under python 3.x, and we officially support python 3.8.

After installing python 3.8, we strongly recommend you to use `virtualenv` (a tool to create isolated Python environments) to manage the python environment. You could use following commands to create an environment `venv` and activate it.

```bash
$ python3.8 -m venv venv
$ source venv/bin/activate
```

## Install TAPEX

The main requirements of our code base is [fairseq](https://github.com/pytorch/fairseq), which may be difficult for beginners to get started in an hour.

However, do not worry, we already wrap all necessary commands for developers.
In other words, you do not need to study fairseq to start your journey about TAPEX!
You can simply run the following command (in the virtual environment) to use TAPEX:

```bash
$ pip install --editable ./
```

> The argument `--editable` is important for your potential follow-up modification on the tapex library. The command will not only install dependencies, but also install `tapex` as a library, which can be imported easily.

## Use TAPEX

Once `tapex` is successfully installed, you could go into [examples](examples) to enjoy fine-tuning TAPEX models and using them on different applications!

# 🏰 Resource

## Pre-training Corpus

Our synthetic pre-training corpus which includes nearly **5,000,000** tuples of (*SQL queries*, *flattened tables*, *SQL execution results*) can be downloaded from [here](https://github.com/microsoft/Table-Pretraining/releases/download/v1.1/tapex_pretrain.tar.gz). You can use it for research purpose, but you should be careful about the [data license](LICENSE-Data).

Below is an example from the pre-training corpus:

- The SQL plus flattened Table as **INPUT**:
```
select vote where passed = 'may 6, 1861' col : state | passed | referendum | vote 
row 1 : s. carolina | december 20, 1860 | none | none 
row 2 : mississippi | january 9, 1861 | none | none row 3 : florida | january 10, 1861 | none | none 
row 4 : alabama | january 11, 1861 | none | none row 5 : georgia | january 19, 1861 | none | none 
row 6 : louisiana | january 26, 1861 | none | none row 7 : texas | february 1, 1861 | february 23 | 46,153-14,747 
row 8 : virginia | april 17, 1861 | may 23 | 132,201-37,451 row 9 : arkansas | may 6, 1861 | none | none 
row 10 : tennessee | may 6, 1861 | june 8 | 104,471-47,183 row 11 : n. carolina | may 20, 1861 | none | none 
row 12 : missouri | october 31, 1861 | none | none row 13 : kentucky | november 20, 1861 | none | none
```
- The SQL Execution Result as **OUTPUT**:
```
104471
```

Here we want to acknowledge the huge effort of paper [On the Potential of Lexico-logical Alignments for Semantic Parsing to SQL Queries](https://arxiv.org/pdf/2010.11246.pdf), which provides the rich resources of SQL templates for us to synthesize the pre-training corpus.
If you are interested, please give a STAR to their [repo](https://github.com/tzshi/squall).

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`tapex.base` | 6 encoder and decoder layers | 140M | [tapex.base.tar.gz](https://github.com/microsoft/Table-Pretraining/releases/download/v1.0/tapex.base.tar.gz)
`tapex.large` | 12 encoder and decoder layers | 400M | [tapex.large.tar.gz](https://github.com/microsoft/Table-Pretraining/releases/download/v1.0/tapex.large.tar.gz)

> More pre-trained models will be uploaded soon!


# 💬 Citation

If our work is useful for you, please consider citing our paper:

```bibtex
@misc{liu2021tapex,
    title={TAPEX: Table Pre-training via Learning a Neural SQL Executor}, 
    author={Qian Liu and Bei Chen and Jiaqi Guo and Zeqi Lin and Jian-guang Lou},
    year={2021},
    eprint={2107.07653},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# 👍 Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# 📝 License

Please note that there are **TWO LICENSES** for code and pre-training corpus.
The code and pre-trained models are open-sourced under [MIT License](LICENSE-Code), while the pre-training corpus is released under [CC BY-SA 4.0](LICENSE-Data).

# ™️ Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
