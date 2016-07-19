#! /bin/bash
cd ./example

count=($(ls -1 ./ | grep .solverstate | wc -l))
filename=$(date +"%F_%H_%M")
echo $filename

gpu_num=0
gpu_count=($(nvidia-smi -L | wc -l))
if [ $gpu_count -gt 1 ]
then
	read -p "You have more than one graphic card. Do you want to see the current process list?(y/n)" answer
	case ${answer:0:1} in
		y|Y )
			nvidia-smi
		;;
	esac
	while :
	do
		read -p "Enter GPU number : " answer
		gpu_num=${answer}
		if [ "$gpu_num" -ge 0 -a "$gpu_num" -lt "$gpu_count" ]
		then
			break
		fi
	done
fi

echo Using GPU '#'$gpu_num.

if [ $count -ge "1" ]
then
	list=($(ls -1 ./*.solverstate | tr '\n' '\0' | xargs -0 -n 1 basename | sort -V -r))
	read -p "You have a solverstate. Do you want to continue learning process from the last(y/n)? " answer
	case ${answer:0:1} in
		y|Y )
			../caffe/build/tools/caffe train -solver ./solver.prototxt -gpu $gpu_num -snapshot ./$list &> $filename.log &
		;;
		* )
			../caffe/build/tools/caffe train -solver ./solver.prototxt -gpu $gpu_num &> $filename.log &
		;;
	esac
else
	../caffe/build/tools/caffe train -solver ./solver.prototxt -gpu $gpu_num &> $filename.log &
fi

cd ..

tail -F ./example/$filename.log

#script for future use

#!/bin/bash
#list=$(ls -1 ./regularized_fix/*.solverstate | tr '\n' '\0' | xargs -0 -n 1 basename)
#for file in $list
#do
#	echo $file
#done
#files=./regularized_fix/"*.solverstate"
#regex='([0-9]+)\.solverstate'
#for f in $files
#do
#	[[ $f =~ $regex ]]
#	echo ${BASH_REMATCH[1]}
#done

#list=$(ls -1 ./regularized_fix/*.solverstate | tr '\n' '\0' | xargs -0 -n 1 basename | sort -V)
