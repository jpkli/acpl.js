acp: acp.o libacp.so testacp

testacp: acp.cpp libacp.so
	g++ -c acp.cpp -I/usr/local/cuda-5.5/include
	g++  acp.o -o testacp -L. -ldl -Wl,-rpath,. -lacp -lcuda -lcudart -L/usr/local/cuda-5.5/lib64  -I/usr/local/cuda-5.5/include

libacp.so : dacp.o
	g++  -shared -Wl,-soname,libacp.so -o libacp.so acp.o dacp.o -lcuda -lcudart -L/usr/local/cuda-5.5/lib64 -I/usr/local/cuda-5.5/include

acp.o : acp.cu acp.hpp
	nvcc -m64 -arch=sm_20 -dc -Xcompiler '-fPIC' acp.cu
	nvcc -m64 -arch=sm_20 -Xcompiler '-fPIC' -dlink acp.o -o dacp.o

#test: link.o libtestthrust.so testmain

#testmain : main.cpp libtestthrust.so
#	g++ -c main.cpp -I/usr/local/cuda-5.5/include
#	g++ -o testmain -L. -ldl -Wl,-rpath,. -ltestthrust -lcuda -lcudart -L/usr/local/cuda-5.5/lib64  main.o -I/usr/local/cuda-5.5/include

#libtestthrust.so : link.o
#	g++  -shared -Wl,-soname,libtestthrust.so -o libtestthrust.so  testthrust.o link.o -lcuda -lcudart -L/usr/local/cuda-5.5/lib64 -I/usr/local/cuda-5.5/include

#link.o : testthrust.cu testthrust.hpp
#	nvcc -m64 -arch=sm_20 -dc -Xcompiler '-fPIC' testthrust.cu
#	nvcc -m64 -arch=sm_20 -Xcompiler '-fPIC' -dlink testthrust.o -o link.o

clean :
	rm -f testmain testthrust.o link.o libtestthrust.so main.o libacp.so acp.o dacp.o libacp.so