# Double Double Dominoes Scorer
In order to run this solution, follow the steps below:
1. Install the dependencies by running
```sh
pip install -r requirements.txt
```
2. In order to generate the aligned images, run
```sh
python align_images.py -r 'board+dominoes/02.jpg' -p <IMAGES_PATH> '-o aligned_train/'
```
(use 'board+dominoes/02.jpg' as the reference filepath as it gives the best results)
3. Open `score_game.ipynb`, set `aligned_images_path` as the path of the output from running the last script and set `out_path` as the desired prediction folder.