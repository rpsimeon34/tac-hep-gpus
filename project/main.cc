#include <iostream>
using namespace std;
const int DSIZE = 10;
const int RADIUS = 3;
const int A_intval = 1;
const int B_intval = 2;


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

void matmul (int A[DSIZE][DSIZE], int B[DSIZE][DSIZE], int (&C)[DSIZE][DSIZE]) {
    for (int i=0; i<DSIZE; i++) {
        for (int j=0; j<DSIZE; j++) {
            int result = 0;
            for (int k=0; k<DSIZE; k++) {
                result += A[i][k]*B[k][j];
            }
            C[i][j] = result;
        }
    }
}

int main () {
    int arr_A[DSIZE][DSIZE];
    int arr_B[DSIZE][DSIZE];
    int sten_A[DSIZE][DSIZE];
    int sten_B[DSIZE][DSIZE];
    int output[DSIZE][DSIZE];
    // Fill in matrices
    for (int m=0; m<DSIZE; m++) {
        for (int n=0; n<DSIZE; n++) {
            arr_A[m][n] = A_intval;
            sten_A[m][n] = arr_A[m][n];
            arr_B[m][n] = B_intval;
            sten_B[m][n] = arr_B[m][n];
            output[m][n] = 0;
        }
    }
    
// Apply stencil, perform multiplication

    stencil(arr_A,sten_A);
    stencil(arr_B,sten_B);
    matmul(sten_A,sten_B,output);

// Check output for accuracy

    int expected_00 = A_intval*B_intval*DSIZE;
    cout << "Output (0,0): " << output[0][0] << ", expected " << expected_00 << endl;
    
    int expected_R10 = A_intval*(RADIUS*4+1)*(DSIZE-2*RADIUS);
    expected_R10 = expected_R10 + 2*RADIUS*A_intval;
    expected_R10 = expected_R10*B_intval;
    cout << "Output (radius+1,0): " << output[RADIUS+1][0] << ", expected " << expected_R10 << endl;

    int expected_0R1 = B_intval*(RADIUS*4+1)*(DSIZE-2*RADIUS);
    expected_0R1 = expected_0R1 + 2*RADIUS*B_intval;
    expected_0R1 = expected_0R1*A_intval;
    cout << "Output (0,radius+1): " << output[0][RADIUS+1] << ", expected " << expected_0R1 << endl;

    int expected_R1R1 = B_intval*A_intval*(RADIUS*4+1)*(RADIUS*4+1)*(DSIZE-2*RADIUS);
    expected_R1R1 = expected_R1R1 + 2*RADIUS*A_intval*B_intval;
    cout << "Output (radius+1,radius+1): " << output[RADIUS+1][RADIUS+1] << ", expected " << expected_R1R1 << endl;

/*
    cout << "Row 1 matrix A after stencil: " << endl;
    for (int i=0; i<DSIZE; i++) {
        cout << sten_A[1][i];
    }
    cout << endl;
    cout << "Column 1 matrix B after stencil: " << endl;
    for (int j=0; j<DSIZE; j++) {
        cout << sten_B[j][1];
    }
    cout << endl;
    cout << "Multiplication output in (1,1): " << endl;
    cout << output[1][1] << endl;
*/
}
