# SVM

for dim in 32 64 128 256 0; do
	matlab -nojvm -r "addpath nips16/experiments/; cmp_svm($dim); exit;"
	mkdir svm-$dim; mv *.mat *.log svm-$dim
done

mkdir svm; mv svm-* svm

# MLP

for dim in 32 64 128 256; do
	PCADIM=$dim python -u nips16/experiments/cmp_mlp.py
	mkdir mlp-$dim; mv *.mat *.log mlp-$dim
done

mkdir mlp; mv mlp-* mlp

# CNN

python -u nips16/experiments/cmp_cnn.py
mkdir cnn; mv *.mat *.log cnn

# VIS

python -u nips16/experiments/vis_filt.py
mkdir vis; mv *.mat vis
