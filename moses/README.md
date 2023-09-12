# The MOSES datasets

The MOSES datasets were taken from https://github.com/molecularsets/moses. Please follow their document to install `molsets` package.

## Training dataset

1,584,663 SMILES were taken from MOSES training dataset and canonicalized using `RDKit` package.

`moses_prop.csv` contains SMILES and penalized logP values of the training dataset.

`moses_smiles.txt` contains SMILES.

`prop.txt` contains penalized logP values.

`vocab.txt` contains vocabulary derived from the training dataset.

Training dataset can be derived from `molsets` package using:

```sh
import moses
train_dataset = moses.get_dataset('train')
```

## Testing dataset

Testing dataset can be devired from `molsets` package using:

```sh
import moses
test_dataset = moses.get_dataset('test')
testsf_dataset = moses.get_dataset('test_scaffolds')
```