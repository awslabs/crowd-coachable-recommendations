import pandas as pd, numpy as np
import requests, tqdm, argparse, os
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='crawl landing images from asin field')
parser.add_argument('--input-file', default='data/prime_pantry/meta_Prime_Pantry.json.gz')
parser.add_argument('--output-file', default="data/prime_pantry/landingImageURL.csv")
parser.add_argument('--output-folder', default="data/prime_pantry/landingImage")
parser.add_argument('--num-retries', type=int, default=200)
args, *_ = parser.parse_known_args()

meta = pd.read_json(args.input_file, lines=True).drop_duplicates('asin').set_index('asin')

# https://localcoder.org/scraping-links-with-beautifulsoup-from-all-pages-in-amazon-results-in-error

headers = {
    'Host': 'www.amazon.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'TE': 'Trailers'
}


# get the urls

landing_url = []
pbar = tqdm.tqdm(meta.index.values)
for asin in pbar:
    try:
        page = requests.get(f"https://www.amazon.com/dp/{asin}", headers=headers)
        soup = BeautifulSoup(page.text, 'html.parser')
        url = soup.findAll("img", {"id": "landingImage"})[0].attrs['src']
        landing_url.append(url)
    except Exception:
        landing_url.append(None)
    pbar.set_description(f"success rate {np.mean([x is not None for x in landing_url])}")

for _ in range(args.num_retries):
    pbar = tqdm.tqdm(np.random.permutation(len(meta)))
    for i in pbar:
        asin = meta.index.values[i]
        if landing_url[i] is None:
            try:
                page = requests.get(f"https://www.amazon.com/dp/{asin}", headers=headers)
                soup = BeautifulSoup(page.text, 'html.parser')
                url = soup.findAll("img", {"id": "landingImage"})[0].attrs['src']
                landing_url[i] = url
            except Exception:
                pass
            pbar.set_description(f"success rate {np.mean([x is not None for x in landing_url])}")

pd.Series(landing_url, index=meta.index.values).to_csv(args.output_file, header=None)


# download the images

os.makedirs(args.output_folder, exist_ok=True)

asin_url = pd.read_csv(args.output_file, names=['asin', 'url']).set_index('asin')['url']

for asin, url in tqdm.tqdm(asin_url.iteritems()):
    if isinstance(url, str):
        suffix = url.split('.')[-1]
        image_file = f'{args.output_folder}/{asin}.{suffix}'
        if not os.path.exists(image_file):
            response = requests.get(url)
            with open(image_file, 'wb') as fp:
                fp.write(response.content)
