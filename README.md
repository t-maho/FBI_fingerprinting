# FBI 

Code to Fingerprint any model with Benign Inputs 

Due to the lack of space, only Top-1 data is released in the GitHub.

Some results of our paper need images. We provide a small set of 1000 images. These images are the ones with the highest entroyp score.



# Run


First, install requirements

```bash
pip install -r requirements
```


## Known

### Detection


```bash
python known_detection.py --input ./data/predictions --score mean_case --family pure --max_drop -0.15
```

family argument can be choosen between: pure, variation, singleton


### Identification


```bash
python known_identification.py --input ./data/predictions --score mean_case --family pure --max_drop -0.15
```


family argument can be choosen between: pure, variation, singleton



## Unknown


### Detection


```bash
python unknown_detection.py --input ./data/predictions/ --output_dir ./results/ --delegate close --family pure --n_images 10 50 100 150 200
```



### Identification Pure

```bash
python unknown_detection.py --input ./data/predictions/ --output_dir ./results/ --delegate close --family pure --n_images 10 50 100 150 200
```




### Identification Variation (with compound)

```bash
python unknown_identification_variation.py --input ./data/predictions/ --output_dir ./results/ --delegate middle close --n_images 50 100 200 --sort_images entropy --n_gen 20
```