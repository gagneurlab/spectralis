# Spectralis

Spectralis is a new method for de novo peptide sequencing that builds upon a new modeling task, bin reclassification. Bin reclassification assigns ion series to discretized m/z values even in the absence of a peak based on amino acid-gapped convolutional layers. 

Spectralis allows the rescoring of any peptide-spectrum match (PSM, Spectralis-score), which can be used as a post-processing step of any existing de novo sequencing tool or to combine results from multiple de novo sequencing tools. Furthermore, Spectralis allows the fine-tuning of peptide-spectrum matches in an evolutionary algorithm (Spectralis-EA).

For more information see:

- Daniela Klaproth-Andrade, Johannes Hingerl, Yanik Bruns, Nicholas H. Smith, Jakob Träuble, Mathias Wilhelm, Julien Gagneur: "Deep learning-driven fragment ion series classification enables highly precise and sensitive de novo peptide sequencing", bioRxiv 2023.01.05.522752; doi: https://doi.org/10.1101/2023.01.05.522752


## Installation

### Prerequisites 

Spectralis was trained and tested using Python 3.7 on a Linux system with a GPU.
The list of required packages for running Spectralis can be found in the file `requirements.txt`.

### Using pip and conda environments

We recommend to install and run Spectralis on a dedicated conda environment. 
To create and activate the conda environment run the following commands:

```
conda create --name spectralis_env python=3.7
conda activate spectralis_env
```

More information on conda environments can be found in Conda's [user guide](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html).

Spectralis requires a PyTorch installation, as indicated in the requirements file `requirements.txt`. However, for running Spectralis with a GPU, we recommend to install PyTorch manually to ensure compatibility between the installation and the user's GPU. For this, check PyTorch's [installation guide](https://pytorch.org/get-started/locally/#start-locally).

To install Spectralis run the following command inside the root directory:

```
pip install .
```



## Getting started

Trained models and example files can be found in the following Zenodo repository:
[zenodo.8393846](https://zenodo.org/record/8393846).

### Configuration

First, create a new configuration file or use the existing file stored in `spectralis_config.yaml`which contains the following features and settings:

- `prosit_ce`: collision energy to be used for collecting Prosit predictions
- `binreclass_model_path`: path to the bin reclassification model
- `num_cores`: number of cores to run Spectralis. Set to -1 to use all available cores.


Settings needed only for Spectralis-score and Spectralis-EA:

- `scorer_path`: path to the model for Spectralis-score
- `max_delta_ppm`: maximal delta difference in ppm to match theoretical to experimental peaks
- `min_intensity`: minimal peak intensity
- `change_prob_thresholds`: probability thresholds to construct numerical features for the scorer from the bin reclassification model. 

Settings needed only for Spectralis-EA:

- `POPULATION_SIZE`: number of individuals in each generation.
- `ELITE_RATIO`: ratio of individuals in a generations that will be considered as elite individuals and passed directly to the next generation.
- `NUM_GEN`: number of generations for the evolutionary algorithm.
- `TEMPERATURE`: temperature constant to compute selection probabilities.
- `MIN_SCORE`: minimal score of input sequences to be fine-tuned. If input sequence has a lower score than `MIN_SCORE`, the initial sequence is returned after Spectralis-EA.
- `MAX_SCORE`: maximal score of input sequences to be fine-tuned. If input sequence has a higher score than `MAX_SCORE`, the initial sequence is returned after Spectralis-EA.
- `bin_prob_threshold`: Minimal probability threshold required for a predicted bin to be considered in the spectrum graph algorithm.
- `input_bin_weight`: Input weight for bins corresponding to initial sequence.

The following setting should be changed only when training a bin reclassification model from scratch. Leave the settings unchanged when using the model from the Zenodo repository.

- `BATCH_SIZE`: GPU batch size
- `BIN_RESOLUTION`: bin resolution 
- `MAX_MZ_BIN`: maximal m/z value to be considered in the model
- `N_CHANNELS`: number of channels in each layer
- `N_CONVS`: number of AA-gapped convolutional layers
- `DROPOUT`: dropout probability
- `KERNEL_SIZE`: kernel size 
- `BATCH_NORM`: indicates whether batch normalization should be applied in each layer
- `ION_TYPES`: list of ion types (e.g. b, y) to be considered by the model
- `ION_CHARGES`: list of ion charges (e.g. singly-charged only) to be considered by the model.
- `ADD_INPUT_TO_END`: indicates whether a skip connection from the input layer to the final layer should be added.
- `add_intensity_diff`: indicates whether an input channel with the intensity differences between theoretical and experimental spectra should be added
- `add_precursor_range`: indicates whether a boolean input channel with the precursor m/z range should be added
- `log_transform`: indicates whether the input intensities should be log-transformed
- `sqrt_transform`: indicates whether the input intensities should be square root-transformed
- `focal_loss`: indicates whether focal loss should be should as the loss function. Otherwise BCE loss will be used.
- `learning_rate`: learning rate for training
- `n_epochs`: number of maximal epochs for training

### Input files

- For running Spectralis-score or Spectralis-EA, a `.csv` or `.mgf` file serves as input. 
- The input file should contain the experimental spectra, precursor charge and m/z, spectrum identifiers, as well as the initial peptide sequence to score or fine-tune. 
- An example of an input file can be found in `example.mgf`.

## Running Spectralis

Spectralis can be run either from the command line or in a Python script.

### Running Spectralis from the command line

Start by testing the Spectralis installation with:

```
spectralis --help
```

#### Spectralis-score

To obtain Spectralis-scores for PSMs in an `.mgf` file, run the following command selecting the rescoring mode (`--mode=rescoring`) :

```
spectralis --mode="rescoring" --input_path="example.mgf" --output_path="output_spectralis_rescoring.csv" --config="spectralis_config.yaml"
```

The computed scores from the input file (`--input_path="<file_name>.mgf"`) will be stored in the specified output file (`--output_path=""<file_name>.csv"`).
If a configuration file is not specified, the default file `spectralis_config.yaml` will be used.

#### Spectralis-EA

Similarly, to fine-tune initial PSMs with Spectralis-EA from an `.mgf` file, run the following command selecting the fine-tuning mode (`--mode=ea`):

```
spectralis --mode="ea" --input_path="example.mgf" --output_path="output_spectralis_ea.csv" --config="spectralis_config.yaml"
```

The fine-tuned sequences together with Spectralis-scores will be stored in the specified output file (`--output_path=""<file_name>.csv"`).


#### Bin reclassification

To get predictions from the bin reclassification mode given an input `.mgf` file, run the following command selecting the bin reclassification mode (`--mode="bin_reclassification"`):

```
spectralis --config="spectralis_config.yaml" --mode="bin_reclassification" --input_path="example.mgf" --output_path="output_binreclass.hdf5"
```


This stores bin probabilities for singly-charged b and y ions with the corresponding m/z bins above the bin probability threshold, as well as the predicted changes and m/z bins for the input sequences in the specified `.hdf5` file.


### Running Spectralis in a Python script

Start running Spectralis by importing the package and creating a `Spectralis` object which takes as input the configuration file:

```
from spectralis.spectralis_master import Spectralis
spectralis = Spectralis(config_path="spectralis_config.yaml")
```


#### Spectralis-score

To obtain Spectralis-scores for PSMs in an `.mgf` file, run the following command:

```
spectralis.rescoring_from_mgf(mgf_path="example.mgf", out_path="spectralis_example_out.csv")
```

The function returns a data frame with Spectralis-scores and spectrum identifiers. The scores can be also stored in an output file specified in the `out_path` argument of the function.

#### Spectralis-EA

To fine-tune initial PSMs with Spectralis-EA from an `.mgf` file, run the following command:

```
spectralis.evo_algorithm_from_mgf(mgf_path="example.mgf", output_path="spectralis-ea_example_out.csv")
```

The function returns a data frame with the Spectralis-scores for initial and fine-tuned sequences for each spectrum identifier.

#### Bin reclassification

Similarly, to get predictions from the bin reclassification mode given an input `.mgf` file, run the following command:

```
binreclass_out = spectralis.bin_reclassification_from_mgf(mgf_path="example.mgf", out_path="output_binreclass.hdf5")
y_probs, y_mz, b_probs, b_mz, y_changes, y_mz_inputs, b_mz_inputs = binreclass_out

```

The function returns bin probabilities for singly-charged b and y ions with the corresponding m/z bins above the bin probability threshold, as well as the predicted changes and m/z bins for the input sequences.

#### Retraining models

With Spectralis, you can train a random forest model regressor or an XGBoost model to estimate the Levenshtein distance of an input to the correct peptide from scratch. For this, you can use the following function: 

```
spectralis.train_scorer_from_csvs(train_paths,          # path containing training data stored in csv file
                                  
                                  # Column names in csv path containing peptide, precursor charge and m/z, 
                                  #     experimental spectra and levenshtein distances
                                  peptide_col, precursor_z_col, exp_mzs_col, exp_ints_col, precursor_mz_col, target_col,
                                  
                                  original_score_col,   # column in csv file indicating original scores from denovo seq tool. Default: None
                                  model_type            # "xgboost" or "rf" 
                                  model_out_path,       # path to store trained model
                                  features_out_dir,     # directory to store feature files
                                  csv_paths_eval        # path to evaluation data
                                  
                                  )
```

## Citation

If you use Spectralis, please cite the following:

- Daniela Klaproth-Andrade, Johannes Hingerl, Yanik Bruns, Nicholas H. Smith, Jakob Träuble, Mathias Wilhelm, Julien Gagneur: "Deep learning-driven fragment ion series classification enables highly precise and sensitive de novo peptide sequencing", bioRxiv 2023.01.05.522752; doi: https://doi.org/10.1101/2023.01.05.522752


## References

- Gessulat S, Schmidt T, Zolg DP, Samaras P, Schnatbaum K, Zerweck J, Knaute T, Rechenberger J, Delanghe B, Huhmer A, Reimer U, Ehrlich HC, Aiche S, Kuster B, Wilhelm M: “PROSIT: Proteome-wide prediction of peptide tandem mass spectra by deep learning”. Nature Methods. 2019; 16(6):509-518. doi: 10.1038/s41592-019-0426-7.
