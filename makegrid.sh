#!/bin/bash

if [ $# -ne 5 ]; then
  echo "usage: $0 <width in markers> <height in markers> <width with unit> <height with unit> <output.pdf>"
  exit 0
fi

echo "start"

let wp=($1+31)/32
let hp=($2+31)/32

let cr=$1*5-1
let cb=$2*5-1

echo "crop $cr $cb"

./hdmarker_generate 0 $wp $hp tmpmakegrid.ppm

pamcut -right $cr -bottom $cb tmpmakegrid.ppm | pnmscale 4 | potrace -o $5 -n -z black -u 1 -t0 -a0 -b pdf -W${3} -H${4}

rm tmpmakegrid.ppm