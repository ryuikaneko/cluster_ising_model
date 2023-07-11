#!/bin/bash

cat ../job/dat_* | grep "#" | head -n 1 > dat
cat ../job/dat_* | grep -v "#" >> dat
