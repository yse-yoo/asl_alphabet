## venv
```bash
python -m venv .venv
```

### Activate
#### Windows
```bash
.venv\Scripts\activate
```

```bash
python -m pip install --upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Mac
```bash
source .venv/bin/activate
```


## Package Install
### Windows
```bash
python -m pip install -r requirements.txt
```

### Mac(Apple Silicon)
```bash
python -m pip install -r requirements_macos.txt
```

### ライブラリ確認
```bash
python -m pip list
```
## ML
### Dataset
#### For training
```bash
asl_words_train/
- I_Love_You/
- Yes/
- No/
- Hello/
- Thank_You/
- Fine/
- Sorry/
- Please/
- Nothing/
```

#### For predict
```bash
custom_test/*.jpg
```

#### Capture
capture images (100)
```bash
python capture_words.py
```

### Training
training by images of asl_words_train/

```bash
python train_words.py
```

### Predict
predict by images of custom_test/
```bash
python predict_words.py
```


## サーバ起動
```bash
uvicorn main:app --reload --port 8000  
```