import os
import splitfolders
# %%
loc = 'dataset'

os.makedirs('split/train')
os.makedirs('split/val')
os.makedirs('split/test')

splitfolders.ratio(loc, output='split',
                   ratio=(0.80, .10, .10))
