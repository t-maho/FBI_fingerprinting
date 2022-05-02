# FBI 

Code to Fingerprint any model with Benign Inputs 

Top-1 data, used in the paper, is in the folder data



# Run


First, install requirements

```bash
pip install -r requirements
```


## Known

### Detection


```bash
python known_detection.py --input ./data --score mean_case --family pure --max_drop -0.15
```

family argument can be choosen between: pure, variation, singleton


### Identification


```bash
python known_identification.py --input ./data --score mean_case --family pure --max_drop -0.15
```


family argument can be choosen between: pure, variation, singleton




## Unknown