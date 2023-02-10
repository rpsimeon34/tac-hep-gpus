#include <iostream>
using namespace std;

string rps (const string &player1, const string &player2) {
    if (player1 == "rock") {
        if (player2 == "rock") {
            return "Tie!";
        } else if (player2 == "paper") {
            return "Player 2 wins!";
        } else if (player2 == "scissors") {
            return "Player 1 wins!";
        }
    } else if (player1 == "paper") {
        if (player2 == "rock") {
            return "Player 1 wins!";
        } else if (player2 == "paper") {
            return "Tie!";
        } else if (player2 == "scissors") {
            return "Player 2 wins!";
        }
    } else if (player1 == "scissors") {
        if (player2 == "rock") {
            return "Player 2 wins!";
        } else if (player2 == "paper") {
            return "Player 1 wins!";
        } else if (player2 == "scissors") {
            return "Tie!";
        }
    } else {
        return "Invalid input";
    }
}

int main () {
    string p1;
    string p2;
    cout << "Player 1: (rock/paper/scissors)" << endl;
    cin >> p1;
    cout << "Player 2: (rock/paper/scissors)" << endl;
    cin >> p2;
    cout << rps(p1,p2) << endl;
}
