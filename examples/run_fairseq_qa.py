from fairseq_cli.train import cli_main as fairseq_train
from fairseq_cli.generate import cli_main as fairseq_generate


def train(processed_data_dir):
    # TODO: set training hyper-parameters
    fairseq_train()


def evaluate(processed_data_dir):
    # TODO: draft an example for evaluation
    pass


def predict(processed_data_dir):
    pass


if __name__ == '__main__':
    pass
