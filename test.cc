#include <iostream>
using namespace std;

int main (){
    int values[3];
    values[0] = 7;
    values[1] = 20;
    values[2] = 54;
    cout << values[0] << ", " << values[1] << "," << values[2] << endl;
    for (int i : values) {
        cout << i << endl;
    }
}
