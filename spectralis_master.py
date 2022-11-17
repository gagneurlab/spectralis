# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = "Daniela Andrade Salazar"
__email__ = "daniela.andrade@tum.de"

Spectralis should provide the following options:

    - Rescoring
    - GA optimization    

"""

from bin_reclassification.peptide2profile import Peptide2Profile
from bin_reclassification.profile2peptide import Profile2Peptide
from bin_reclassification.models import P2PNetPadded2dConv

from lev_scoring.scorer import PSMLevScorer

class Spectralis():
    
    def __init__(self, config_path,
                 ):
        
        self.config = yaml.load(open(config_path), Loader=yaml.FullLoader) # load model params   
        self.verbose = self.config['verbose']
        
        self.prosit_predictor = self._init_prosit_predictor()
        print(f'[INFO] Initiated prosit predictor')
        
        self.binreclass_model = self._init_binreclass_model()
        print(f'[INFO] Loaded bin reclass P2P-model')
        
        self.peptide2profiler = self._init_peptide2profile()
        self.profile2peptider = self._init_profile2peptide()
        print(f'[INFO] Initiated P2P objects')
        
        self.bin_reclassifier = self._init_binreclassifier()
        print(f'[INFO] Initiated bin reclassifier')
        
        self.scorer = self._init_scorer()  
        print(f'[INFO] Initiated lev scorer')
        
    def _init_binreclassifier(self):
        return BinReclassifier( peptide2profiler=self.peptide2profiler,
                 batch_size=self.config['BATCH_SIZE'],
                 min_bin_change_threshold=min(self.config['change_prob_thresholds']), ## check that this works
                 min_bin_prob_threshold=self.config['bin_prob_threshold']
                 )
    
    def _init_scorer(self):
        return PSMLevScorer(self.config['scorer_path'], 
                             self.config['change_prob_thresholds'],
                             self.config['min_intensity']
                            )
    
    def _init_prosit_predictor(self):
        return PROSITpredictor(server=self.config['server'],
                                path_to_ca_certificate = self.config['path_to_ca_certificate'],
                                path_to_key_certificate = self.config['path_to_key_certificate'],
                                path_to_certificate =  self.config['path_to_certificate']
                              )
    
    def _init_binreclass_model(self,  num=0):
        if torch.cuda.is_available():
            torch.cuda.set_device(num)
        self.device = torch.device(f'cuda:{num}' if torch.cuda.is_available() else 'cpu')
        
        
        in_channels = len(self.config['ION_CHARGES'])*len(self.config['ION_TYPES'])+2
        in_channels = in_channels+2 if self.config['add_intensity_diff'] else in_channels
        in_channels = in_channels+1 if self.config['add_precursor_range'] else in_channels
    
        model = P2PNetPadded2dConv(num_bins=self.config['BIN_RESOLUTION']*self.config['MAX_MZ_BIN'],
                                               in_channels=in_channels,
                                               hidden_channels=self.config['N_CHANNELS'],
                                               out_channels=2,
                                               num_convs=self.config['N_CONVS'], 
                                               dropout=self.config['DROPOUT'],
                                               bin_resolution=self.config['BIN_RESOLUTION'],
                                               batch_norm=self.config['BATCH_NORM'],
                                               kernel_size=(3, self.config['KERNEL_SIZE']), 
                                               padding=(1, 0 if self.config['KERNEL_SIZE']==1 else 1),
                                               add_input_to_end=self.config['ADD_INPUT_TO_END']
                                            )
        checkpoint = torch.load(self.config['binreclass_model_path'], map_location=self.device)
        new_checkpoint = dict()
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                new_checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
            else:
                new_checkpoint[key] = checkpoint[key]
        model.load_state_dict(new_checkpoint)

        if str(self.device) != 'cpu':
            model.cuda()
        model.eval()        
        
        return model
    
    def _init_peptide2profile(self):
        return Peptide2Profile(bin_resolution=self.config['BIN_RESOLUTION'],
                               max_mz_bin=self.config['MAX_MZ_BIN'], 
                               considered_ion_types=self.config['ION_TYPES'], 
                               considered_charges=self.config['ION_CHARGES'],
                               add_leftmost_rightmost=self.config['add_leftmost_rightmost'],
                               verbose=self.verbose,
                               prosit_predictor=self.prosit_predictor,
                               sqrt_transform=self.config['sqrt_transform'],
                               log_transform=self.config['log_transform']
                             )
    
    def _init_profile2peptide(self):
        return Profile2Peptide(  bin_resolution=self.config['BIN_RESOLUTION'], 
                                 max_mz_bin=self.config['MAX_MZ_BIN'], 
                                 prob_threshold=self.config['bin_prob_threshold'],
                                 input_weight = self.config['input_bin_weight'],
                                 verbose=self.verbose,
                               )

    def genetic_algorithm(self, seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities, prosit_ce, out_dir):
        optimizer = GAOptimizer(bin_reclassifier=self.bin_reclassifier, 
                                 prosit_predictor=self.prosit_predictor, 
                                 profile2peptider=self.profile2peptider,
                                 scorer=self.scorer,

                                 lookup_table_path=self.config['aa_mass_lookup_path'],
                                 max_delta_ppm=self.config['max_delta_ppm'], 
                                 population_size=self.config['POPULATION_SIZE'], 
                                 elite_ratio=self.config['ELITE_RATIO'],
                                 n_generations=self.config['NUM_GEN'], 
                                 selection_temperature=self.config['TEMPERATURE'], 
                                 prosit_ce=prosit_ce, 
                                 min_intensity=self.config['min_intensity'],
                                 max_score_thres=self.config['MAX_SCORE'],
                                 min_score_thres=self.config['MIN_SCORE'],

                                 out_dir=out_dir,
                                 write_pop_to_file=self.config['write_pop_to_file'],
                                 num_cores=self.config['num_cores'],
                                 with_cache=self.config['cache_scores'], 
                                 verbose=self.verbose, 
                            )
        
        optimizer.run_optimization(seqs, precursor_z, precursor_m, scans, exp_mzs, exp_intensities)
        
    
    
    def rescoring_from_csv(csv_path, seq_col, charge_col, exp_mzs_col, exp_ints_col, prosit_ce):
        ### TODO
        return 
        
    
    def rescoring(self, seqs, charges, prosit_ce, exp_ints, exp_mzs, precursor_mzs):
        
        self.prosit_predictor.predict(sequences=seqs, 
                                      charges=[int(c) for c in charges], 
                                      collision_energies=[prosit_ce]*len(seqs), 
                                      models=["Prosit_2019_intensity"])['Prosit_2019_intensity']
        prosit_mzs, prosit_ints, prosit_anno =  prosit_out['fragmentmz'],prosit_out['intensity'], prosit_out['annotation']
        
        peptide_masses = np.array([U._compute_peptide_mass_from_seq(seqs[j]) for j in range(len(seqs)) ])
        binreclass_out = self.bin_reclassifier.get_binreclass_preds(prosit_mzs=prosit_mzs,
                                                          prosit_ints=prosit_ints,
                                                          prosit_anno=prosit_anno, 
                                                          pepmass=peptide_masses,
                                                          exp_mzs=exp_ints,
                                                          exp_int=exp_mzs,
                                                          precursor_mz=precursor_mzs,
                                                        )
        y_probs, y_mz_probs, b_probs, b_mz_probs, y_changes, y_mz_inputs, b_mz_inputs = binreclass_out
        
        return self.scorer.get_scores(exp_mzs, exp_ints, prosit_ints, prosit_mzs, y_changes)
        