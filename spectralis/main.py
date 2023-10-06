### Command line entry point for Spectralis
import logging
logging.captureWarnings(True)

import click
from typing import Optional

from .spectralis_master import Spectralis


@click.command()
@click.option(
    "--mode",
    required=True,
    default="rescoring",
    help="""\b\nOptions to run Spectralis:\n
    - "rescoring" will predict confidence scores for peptide-spectrum matches (PSMs).\n
    - "ea" will fine-tune and provide confidence scores for PSMs\nin an evolutionaty algorihtm.\n
    - "bin_reclassification" will provide bin probability predictions for ions of input PSMs.""",
    type=click.Choice(["rescoring", "ea", "bin_reclassification"]),
)
@click.option(
    "--config",
    default='../spectralis_config.yaml',
    help="Configuration file containing Spectralis custom options. Default configuration file: https://github.com/gagneurlab/spectralis/blob/main/spectralis_config.yaml.",
    type=click.Path(exists=True, dir_okay=False),
)


@click.option(
    "--input_path",
    help="MGF input file containing PSMs. See example here: example.mgf",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output_path",
    help="Output path to store Spectralis' output.",
    type=click.Path(dir_okay=False),
)

def main(   mode,
            input_path,
            config,
            output_path,
        ):
             
    '''
    ============= SPECTRALIS =============================
    
    Spectralis is new method for de novo peptide sequencing 
    which builds upon a new modeling task, bin reclassification, 
    which assigns ion series to discretized m/z values 
    even in the absence of a peak based on amino acid-gapped 
    convolutional layers.\n
    
    Spectralis allows:\n
    
    - the rescoring of any peptide-spectrum match (PSM, Spectralis-score)\n
    - fine-tuning of peptide-spectrum matches in an evolutionary algorithm (Spectralis-EA)\n
    - bin reclassification for collecting probabilities of bins to posses ions of a certain type (e.g. b+, y+)\n
        
        
    Daniela Klaproth-Andrade, Johannes Hingerl, Nicholas H. Yanik Bruns, Smith, Jakob Träuble, Mathias Wilhelm, Julien Gagneur: \n
        
    "Deep learning-driven fragment ion series classification enables highly precise and sensitive de novo peptide sequencing",\n
    
    bioRxiv 2023.01.05.522752; \n
    
    doi: https://doi.org/10.1101/2023.01.05.522752\n
    
    ==============================================================
    '''
        
    print('=================== SPECTRALIS ===================\n\n\n')
    
    ## Initialize Spectralis object reading config file
    spectralis = Spectralis(config)
    
    if mode=="rescoring":
        print('\t-- Rescoring for PSM confidence scores')
        spectralis.rescoring_from_mgf(mgf_path=input_path, out_path=output_path)
        
    elif mode=='ea':
        print('\t-- Evolutionary algorithm for fine-tuning PSMs')
        spectralis.evo_algorithm_from_mgf(mgf_path=input_path, output_path=output_path)
        
    elif mode=='bin_reclassification':
        print('\t-- Bin reclassification for bin probabilities of ions')
        spectralis.bin_reclassification_from_mgf(mgf_path=input_path, out_path=output_path)
        
    else:
        print('\t-- Mode not supported yet')
    
    
    
if __name__ == "__main__":
    main()