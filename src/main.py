#!/bin/env python3

from preproc import Preprocessor, LangEnum
from ui import get_cli_parser

if __name__ == '__main__':
    parser = get_cli_parser()
    # args = parser.parse_args()

    # files = get_files(args['input_dir'])

    preprocessor = Preprocessor(language=LangEnum.EN,
                                stop_words=None)
    texts, dictionary = preprocessor.preprocess_texts(
        [
            'Bleep-bloop, I am a robot!',
            'There is a number (12345) and an email (hello-there@box.com).'
        ]
    )
    print(set(dictionary.values()))

    # Do actual stuff…
