#!/bin/bash

while read file;
do
	echo "Processing $file file..."
	python plot_spectra1D_gen.py $file
done <masklist.txt
