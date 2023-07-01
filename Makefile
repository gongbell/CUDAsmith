NVCC=nvcc
LIBS= -lcuda
all:cuda_launcher
cuda_launcher:cuda_launcher.cu
	$(NVCC) -rdc=true $(LIBS) -Xptxas -O0 -o test cuda_launcher.cu  -w -arch sm_50
#cuda_launcher:cuda_launcher.c
#	$(NVCC) -rdc=true $(LIBS) -o cuda_launcher cuda_launcher.c
clean:
	rm -rf test 
rebuild:clean all
   

