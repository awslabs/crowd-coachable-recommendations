We'll start by loading the Prime Pantry reviews dataset. You will need to fill out the form for access to the data files:

http://deepyeti.ucsd.edu/jianmo/amazon/index.html

Citation:

> Justifying recommendations using distantly-labeled reviews and fined-grained aspects
> Jianmo Ni, Jiacheng Li, Julian McAuley
> Empirical Methods in Natural Language Processing (EMNLP), 2019

```
!wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Prime_Pantry.json.gz
```

We also include a collection of human responses for ground-truth similarities between items. The candidate items are produced from four distinct baseline algorithms: TF-IDF (zero-shot), Bert (zero-shot), Behavioral (supervised by co-review histories of the same users), and Random.

```
!gunzip < prime_pantry_test_response.json.gz | less
```

We also include a list of urls for product images. These are not immediately required for offline training and evaluation. However, you may add the images in a `landingImage` folder for visualization and crowd-sourcing task designs using `ccrec.env.i2i_env.I2IImageEnv` class.

```
!gunzip < landingImageURL.csv.gz | less
```

Finally, we provide a utility function `ccrec.util.amazon_review_prime_pantry.get_item_df` to extract a standard `item_df` table. The table may be fed into zero-shot training functions such as `ccrec.models.vae_lightning.vae_main`. See `test/test_ccrec.py` for some similar examples used in automated tests.
