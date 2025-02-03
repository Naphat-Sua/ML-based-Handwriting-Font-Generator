ML_Handwriting_Font_Generator/
│
├── data/
│   ├── raw/
│   │   └── handwriting_samples/
│   │       ├── user1/
│   │       │   ├── sample1.png
│   │       │   ├── sample2.png
│   │       │   └── ...
│   │       └── user2/
│   │           ├── sample1.png
│   │           ├── sample2.png
│   │           └── ...
│   └── processed/
│       ├── images/
│       └── labels/
│
├── models/
│   ├── cnn_model.h5
│   ├── gan_model.h5
│   └── ...
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── font_generation.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── font_generation.py
│   ├── evaluation.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── app.py
