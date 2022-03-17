To produce results of class examples, please run the following commands.

- train a model:
python recognition.py

- evaluate the trained model:
python recognition.py --mode test

- calculate entropy on the test set:
python cal_entropy.py --n_bins 256

- calculate entropy of three images:
python cal_entropy.py --mode single --n_bins 256 --im_path data/images/1647.jpg
python cal_entropy.py --mode single --n_bins 256 --im_path data/images/1605.jpg
python cal_entropy.py --mode single --n_bins 256 --im_path data/my_images/a.png --visualize

Note: the option --visualize is to show fe



