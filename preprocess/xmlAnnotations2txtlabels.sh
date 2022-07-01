Annotations=$1
dist=$2

i=0
for xml in $Annotations/*;
do
	echo "#$i"
	chmod 744 $xml
	width=$(cat $xml | grep width | sed -e 's/^.*>\(.*\)<\/.*$/\1/g')
	height=$(cat $xml | grep height | sed -e 's/^.*>\(.*\)<\/.*$/\1/g')
	cat $xml | grep xmin | sed -e 's/^.*>\(.*\)<\/.*$/\1/g' > ./tmpx1
	cat $xml | grep xmax | sed -e 's/^.*>\(.*\)<\/.*$/\1/g' > ./tmpx2
	cat $xml | grep ymin | sed -e 's/^.*>\(.*\)<\/.*$/\1/g' > ./tmpy1
	cat $xml | grep ymax | sed -e 's/^.*>\(.*\)<\/.*$/\1/g' > ./tmpy2
	paste -d , ./tmpx1 ./tmpx2 ./tmpy1 ./tmpy2 > ./xxyy

	distfile=$dist/`basename $xml .xml`.txt
	# echo $distfile
	while read row;
	do
		c1=`echo ${row} | cut -d , -f 1`
		c2=`echo ${row} | cut -d , -f 2`
		x=`echo "scale=6; ($c1 + $c2) / $width / 2.0" | bc -l`
		w=`echo "scale=6; ($c2 - $c1) / $width" | bc -l`
		c3=`echo ${row} | cut -d , -f 3`
		c4=`echo ${row} | cut -d , -f 4`
		y=`echo "scale=6; ($c3 + $c4) / $height / 2.0" | bc -l`
		h=`echo "scale=6; ($c4 - $c3) / $height" | bc -l`
		# echo $x
		# echo $w
		# echo $y
		# echo $h
		echo "0 $x $y $w $h" >> $distfile
	done < ./xxyy
	i=$((i+1))
	echo "-> done"
done
rm ./tmpx1 ./tmpx2 ./tmpy1 ./tmpy2 ./xxyy
