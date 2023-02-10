#include <iostream>
using namespace std;

void swap (int &a, int &b) {
    int swapper;
    swapper = a;
    a = b;
    b = swapper;
}

int main () {
    int A[10] = {8,720,6,11,60,103,-104,42,-2,99};
    int B[10] = {1,-2,3,-4,5,-6,7,-8,9,10};
    for (int n=0; n<10; n++) {
        swap(A[n],B[n]);
    }
    cout << "A: " << A[0];
    for (int m=1; m<10; m++) {
        cout << ", " << A[m];
    }
    cout << endl;
    cout << "B: " << B[0];
    for (int k=1; k<10; k++) {
        cout << ", " << B[k];
    }
    cout << endl;
}
