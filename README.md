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

## Kaggle
```bash
pip install kaggle
```

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

```bash
unzip asl-alphabet.zip -d asl_alphabet
```

## Mac TensorFlow Metal
```bash
pip uninstall tensorflow
pip install tensorflow-macos
pip install tensorflow-metal
```

## サーバ起動
```bash
uvicorn app:app --reload --port 8000  
```