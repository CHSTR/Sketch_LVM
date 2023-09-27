import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='LN_prompt')

# --------------------
# DataLoader Options
# --------------------

# Path to 'Sketchy' folder holding Sketch_extended dataset. It should have 2 folders named 'sketch' and 'photo'.
parser.add_argument('--data_dir', type=str, default='/media/chr/Nuevo vol/ecommerce/Flickr15K_dataset') #/home/chr/Sketch_LVM/dataset/Sketchy 
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# Add a ner argument for read .txt file
parser.add_argument('--txt_train', type=str, default='/home/chr/Sketch_LVM/dir') #/home/chr/Sketch_LVM/dir
parser.add_argument('--txt_test', type=str, default='/home/chr/Sketch_LVM/dir') #/home/chr/Sketch_LVM/dir

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--workers', type=int, default=128)
parser.add_argument('--model_type', type=str, default='two_encoder', choices=['one_encoder', 'two_encoder'])

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

# ----------------------
# SBIR Parameters
# ----------------------
parser.add_argument('--model', type=str, default="two_encoders_067_ecommerce_pidinet.ckpt")
parser.add_argument('--output_file', type=str, default="two_encoders_067_ecommerce_pidinet")
parser.add_argument('--image_file', type=str, default="/home/chr/Sketch_LVM_ecommerce/dir/image_test.txt") #/home/chr/Sketch_LVM_ecommerce/dir/dataset_2009/photos_2009.txt # /home/chr/Sketch_LVM_ecommerce/dir/image_test.txt #/home/chr/Sketch_LVM_ecommerce/dir/flickr15/dataset.txt
parser.add_argument('--sketch_file', type=str, default="/media/chr/Nuevo vol/ecommerce/ecommerce/sketches_valid/sk_classification.txt")#/home/chr/Sketch_LVM_ecommerce/dir/dataset_2009/sketches_2009.txt #/media/chr/Nuevo vol/ecommerce/ecommerce/sketches_valid/sk_classification.txt #/home/chr/Sketch_LVM_ecommerce/dir/flickr15/query_class.txt

opts = parser.parse_args()
