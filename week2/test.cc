#include <iostream>
using namespace std;

struct Person {
    string age;
    float weight;
};

void print(Person $person) {
    cout << person.age << endl;
}

int main (){
    Person charis;
        charis.age = "29";
        charis.weight = 55.5;
    print(charis);
//    int values[3];
//    values[0] = 7;
//    values[1] = 20;
//    values[2] = 54;
//    cout << values[0] << ", " << values[1] << "," << values[2] << endl;
//    for (int i : values) {
//        cout << i << endl;
//    }
}
