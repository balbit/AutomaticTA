# NLP Project: Expert-Student LLMs

By Alex and Elliot

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install llamaindex.

```
pip install llama-index
```

Then export the OPENAI key:

```
export OPENAI_API_KEY=XXXXX
```

## Usage

Important functions are currently in utils.py and prompting.py

### Adding Documents

To add documents to the public dataset, add them to the directory data/public_new and data/private_new (text and pdf are both supported!)

Then run `update_public_data(True)` and `update_private_data(True)` in utils.py. 

This should automatically move the files to from public_new to public_data after storing the embeddings in public_persist.

If something breaks or you want to train with a different set of data (eg for another class), move all desired files to public_new/private_new and run `update_public_data(False)` `update_private_data(False)` to reindex everything.

### Running the Tester

To recreate our testing environment, first

1. Copy relevant context files to data/public_new and data/private_new. Class materials we used (except for 6.009 because of uploading difficulties) can be found in https://github.com/heale04/AutoTADATA
2. Add documents by running the script
3. Change the test_class variable to the class name prefix of your new batch of data
4. run `python3 prompting.py` in the root folder!

## License

This project is licensed under the [MIT License](LICENSE).

