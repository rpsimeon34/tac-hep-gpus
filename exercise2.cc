#include <iostream>
using namespace std;

struct Student {
    string name;
    string email;
    string username;
    string experiment;
};

void print(const Student &x) {
    cout << "Name: " << x.name << endl;
    cout << "Email address: " << x.email << endl;
    cout << "Username: " << x.username << endl;
    cout << "Experiment: " << x.experiment << endl;
}

int main () {
    Student Ryan;
        Ryan.name = "Ryan Simeon";
        Ryan.email = "rsimeon@wisc.edu";
        Ryan.username = "rpsimeon34";
        Ryan.experiment = "CMS";
    print(Ryan);
    cout << endl;
    Student Trevor;
        Trevor.name = "Trevor Nelson";
        Trevor.email = "twnelson2@wisc.edu";
        Trevor.username = "twnelson0";
        Trevor.experiment = "CMS";
    print(Trevor);
    cout << endl;
    Student Elise;
        Elise.name = "Elise Chavez";
        Elise.email = "emchavez@wisc.edu";
        Elise.username = "Nanoemc";
        Elise.experiment = "CMS";
    print(Elise);
    cout << endl;
    Student Stephanie;
        Stephanie.name = "Stephanie Kwan";
        Stephanie.email = "skwan@princeton.edu";
        Stephanie.username = "skkwan";
        Stephanie.experiment = "CMS";
    print(Stephanie);
}
