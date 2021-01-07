"""CLI interface using argparse.ArgumentParser"""

from argparse import ArgumentParser
from os import PathLike
from os.path import join as path_join, isfile
from glob import iglob


def get_cli_parser():
    parser = ArgumentParser(description='Topic modelling utility')

    parser.add_argument(
        'input_dir',
        help='Directory where input text files are located.',
        type=str
    )
    parser.add_argument(
        '-m', '--model',
        help='Path to pretrained topic model',
        type=str
    )
    parser.add_argument(
        '-i', '--infer',
        help='Use specified model to infer on new dataset',
        type=str
    )
    parser.add_argument(
        '-t', '--type',
        help='"lda", "nmf", "bert" or "topic2vec"',
        default='lda',
        choices=['lda', 'nmf', 'bert', 'topic2vec'],
        type=str
    )
    parser.add_argument(
        '-s', '--stopwords',
        help='Path to text file containing stopwords to add',
        default=None,
        type=str
    )
    return parser


def get_files(path: PathLike):
    files = iglob(path_join(path, '**/*.txt'), recursive=True)
    for file in files:
        if isfile(file):
            yield file
