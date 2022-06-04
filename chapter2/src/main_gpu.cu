# include <stdio.h>

__global__ void kernel(void){
    printf("hello world gpu\n.");
}

int main(void){
    
    printf("Hello, World cpu!\n");
    kernel<<<256,1>>>();
    return 0;
}
