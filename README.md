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

Important functions are currently in test.py

### Adding Documents

To add documents to the public dataset, add them to the directory public_new (only text is supported right now)

Then run `update_public_data(True)` in test.py. This should automatically move the files to from public_new to public_data after storing the embeddings in public_persist.

If something breaks, move all files to public_new and run `update_public_data(False)` to reindex everything.

### Querying

Example of a query:
```
public_index = index_from_dir('./data/public_persist')
test_query(public_index, 'Summarize the example of Slutsky\'s theorem that is given. Which lecture (1-3) is it from?')
test_query(public_index, 'What is the question being answered by the kiss experiment?')
```

## License

This project is licensed under the [MIT License](LICENSE).

