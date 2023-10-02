from setuptools import setup,find_packages

setup(
        name='spectralis',
        version='2.0',
        author="Daniela Klaproth-Andrade", 
        author_email = "daniela.andrade@tum.de",
        description = 
        '''Spectralis is new method for de novo peptide sequencing which builds upon a new modeling task,
             bin reclassification, which assigns ion series to discretized m/z values 
             even in the absence of a peak based on amino acid-gapped convolutional layers.
             Spectralis allows:
                - the rescoring of any peptide-spectrum match (PSM, Spectralis-score)
                - fine-tuning of peptide-spectrum matches in an evolutionary algorithm (Spectralis-EA)
                - bin reclassification for collecting probabilities of bins to posses 
                    ions of a certain type (e.g. b+, y+)
        ''',
        python_requires = ">=3.7",
        install_requires = ['click',
                             'editdistance==0.5.3',
                             'h5py',
                             'fastparquet',
                             'grpcio', 
                             'networkx',
                             'numpy ',
                             'pandas==1.3.5',
                             'pyaml',
                             'pyteomics==4.6',
                             'scikit-learn',
                             'scipy==1.6.1',
                             'torch',
                             'tqdm',
                             'tritonclient[all]',
                             'xgboost==1.6.2'],
        url='https://github.com/gagneurlab/spectralis',
        license='',
        packages=find_packages(include=['spectralis', 'spectralis.*', 'data']), #['spectralis'],
        zip_safe=False,
        entry_points={
            'console_scripts': [
                'spectralis = spectralis.main:main',
            ],
        },
    include_package_data=True,
    package_data={'spectralis': ['data/*.csv']},
         
         
)