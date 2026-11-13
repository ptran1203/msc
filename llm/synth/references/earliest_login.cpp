#include <iostream>
#include <string>
using namespace std;
int main(){
    string f, l;
    cin >> f >> l;
    string best;
    bool first = true;
    for (size_t i = 1; i <= f.size(); i++) {
        for (size_t j = 1; j <= l.size(); j++) {
            string cand = f.substr(0, i) + l.substr(0, j);
            if (first || cand < best) { best = cand; first = false; }
        }
    }
    cout << best << endl;
}
