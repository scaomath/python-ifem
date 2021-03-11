# πfem
πfem (Pi-FEM or Python-iFEM) is a compact Python finite element library modified based on the popular MATLAB package [iFEM](https://github.com/lyc102/ifem) by Dr. Long Chen. 

The main core of this package is ported from iFEM, and is largely readable extremely suitable for pedagogical purposes. The procedures are mostly sequential in nature like C, except a few classes on mesh objects to make the codes cleaner. The original iFEM package has remnants of being developed over a decade by student researchers, and the MATLAB's clunky indexing, so here I tried to make the code even more readable. Overall this package avoids the modern object-oriented modular Pythonic style.


# Requirements
```bash
$ cat requirements.txt 
matplotlib==3.3.4
numpy==1.19.2
seaborn==0.11.1
pandas==1.2.3
torch==1.8.0
scipy==1.6.1
psutil==5.8.0
jupyterthemes==0.20.0
plotly==4.14.3
ipython==7.21.0

```


# Repo structure

```bash
├── libs
│   ├── mesh generation
│   ├── aux structures
│   ├── fem.py: equations
│   └── ...
├── examples
│   ├── finite element example
│   └── pde data
├── tests
│   └── playground
├── utils.py: utility functions
├── README.md: logs
└── .gitignore
```

## Acknowledgements
The author is grateful to Dr. Long Chen at UC Irvine for the guidance, and appreciates the NSF for the partial support by DMS-1913080.
