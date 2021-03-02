# πfem
πfem (Pi-FEM or Python-iFEM) is a compact Python finite element library based on the popular MATLAB package [iFEM](https://github.com/lyc102/ifem). The main core of this package is ported from iFEM, and is largely readable extremely suitable for pedagogical purposes. The procedure is sequential in nature, instead of embracing the modern obj-oriented modular style.


# Repo structure

```bash
├── libs
│   ├── mesh generation
│   └── auxiliary structure
├── examples
│   ├── finite element example
│   └── pde data
├── equation
│   ├── matrix assembly and solver
├── tests
│   ├── playground
├── utils.py: utility functions
├── README.md: logs
└── .gitignore
```

## Acknowledgements
The author is grateful to Dr. Long Chen at UC Irvine for the guidance, and appreciates the NSF for the partial support by DMS-1913080.
