#!/bin/bash
PWDDIR=`pwd`


jpx400_nikkei225_500_split() {
    # jpx400_nikkei225_500.txt分割する
    TXT=jpx400_nikkei225_500.txt

    OUT_DIR=jpx400_nikkei225_500_split
    mkdir -p ${OUT_DIR}

    sed -n 1,100p ${TXT} > ${OUT_DIR}/1-100.txt
    sed -n 101,200p ${TXT} > ${OUT_DIR}/101-200.txt
    sed -n 201,300p ${TXT} > ${OUT_DIR}/201-300.txt
    sed -n 301,400p ${TXT} > ${OUT_DIR}/301-400.txt
    sed -n 401,500p ${TXT} > ${OUT_DIR}/401-500.txt
    sed -n 501,561p ${TXT} > ${OUT_DIR}/501-561.txt
}

nikkei1000_split() {
    # nikkei1000.txt分割する
    TXT=nikkei1000.txt

    OUT_DIR=nikkei1000_split
    mkdir -p ${OUT_DIR}

    sed -n 1,100p ${TXT} > ${OUT_DIR}/1-100.txt
    sed -n 101,200p ${TXT} > ${OUT_DIR}/101-200.txt
    sed -n 201,300p ${TXT} > ${OUT_DIR}/201-300.txt
    sed -n 301,400p ${TXT} > ${OUT_DIR}/301-400.txt
    sed -n 401,500p ${TXT} > ${OUT_DIR}/401-500.txt
    sed -n 501,600p ${TXT} > ${OUT_DIR}/501-600.txt
    sed -n 601,700p ${TXT} > ${OUT_DIR}/601-700.txt
    sed -n 701,800p ${TXT} > ${OUT_DIR}/701-800.txt
    sed -n 801,900p ${TXT} > ${OUT_DIR}/801-900.txt
    sed -n 901,994p ${TXT} > ${OUT_DIR}/901-994.txt
}

#jpx400_nikkei225_500_split;
nikkei1000_split;