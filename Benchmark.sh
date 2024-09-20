#!/bin/bash

touch benchresults.txt

printf "std, negative, no scaling\n\nCPU:\n\n" >> benchresults.txt

for i in {1..$1};
do sofiaGPU sofia.par scfind.statistic=std pipeline.useGPU=false pipeline.benchmark=true;
done;

printf "\nGPU:\n\n" >> benchresults.txt
for i in {1..$1}
do 
	sofiaGPU sofia.par scfind.statistic=std pipeline.useGPU=true pipeline.benchmark=true
done;

printf "" >> benchresults.txt

printf "\nmad, negative, no scaling\n\nCPU:\n\n" >> benchresults.txt
for i in {1..$1}
do
	sofiaGPU sofia.par scfind.statistic=mad pipeline.useGPU=false pipeline.benchmark=true
done

printf "\nGPU:\n\n" >> benchresults.txt

for i in {1..$1}
do
	sofiaGPU sofia.par scfind.statistic=mad pipeline.useGPU=true pipeline.benchmark=true
done

printf "" >> benchresults.txt

printf "\ngauss, negative, no scaling\n\nCPU:\n\n" >> benchresults.txt
for i in {1..$1}
do
	sofiaGPU sofia.par scfind.statistic=gauss pipeline.useGPU=false pipeline.benchmark=true
done
printf "\nGPU:\n\n" >> benchresults.txt
for i in {1..$1}
do
	sofiaGPU sofia.par scfind.statistic=gauss pipeline.useGPU=true pipeline.benchmark=true
done
