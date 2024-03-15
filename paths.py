# Get paths to the data and results directories.
import os 
src_path = os.path.dirname(os.path.realpath(__file__))
assignment3_path = os.path.dirname(src_path)
data_dir = os.path.join(assignment3_path,'data')
model_dir = os.path.join(assignment3_path,'model')
results_dir = os.path.join(assignment3_path,'results')
