wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip

unzip -j \*.zip && rm *.zip
python amat2pkl.py && rm *.amat
