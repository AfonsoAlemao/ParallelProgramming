./ispc --arch=x86-64 --target=avx2-i32x8 iterative_sort_ispc.ispc -o iterative_sort_ispc.o -h iterative_sort_ispc.h
g++ -O3 -mavx -c -std=c++0x iterative_sort_avx.cpp -o iterative_sort_avx.o
g++ -O0 -std=c++0x -m64 -c iterative_sort.cpp -o iterative_sort.o
g++ -pthread -std=c++0x -m64 iterative_sort.o iterative_sort_ispc.o iterative_sort_avx.o -o iterative_sort

