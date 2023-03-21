#include <iostream>
using namespace std;
const int DSIZE = 10;
const int RADIUS = 3;


void stencil (int arr[DSIZE][DSIZE], int (&out)[DSIZE][DSIZE]) {
    for (int m=RADIUS; m<DSIZE-RADIUS; m++) {
        for (int n=RADIUS; n<DSIZE-RADIUS; n++) {
            int result = arr[m][n];
            for (int k=1; k<RADIUS+1; k++) {
                result += arr[m-k][n];
                result += arr[m+k][n];
                result += arr[m][n-k];
                result += arr[m][n+k];
            }
            out[m][n] = result;
        }
    }
}

int main () {
    int arr_A[DSIZE][DSIZE];
    int arr_B[DSIZE][DSIZE];
    int sten_A[DSIZE][DSIZE];
    int sten_B[DSIZE][DSIZE];
    for (int m=0; m<DSIZE; m++) {
        for (int n=0; n<DSIZE; n++) {
            arr_A[m][n] = 1;
            sten_A[m][n] = arr_A[m][n];
            arr_B[m][n] = 2;
            sten_B[m][n] = arr_B[m][n];
        }
    }
    
    stencil(arr_A,sten_A);
    stencil(arr_B,sten_B);
    cout << sten_A[1][1] << endl;
    cout << sten_B[5][5] << endl;
}
