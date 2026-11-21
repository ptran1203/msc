#include <iostream>
#include <cctype>
using namespace std;
int main(){
    string s;
    getline(cin, s);
    string t;
    for (char c : s) {
        if (isalnum((unsigned char)c)) t += tolower((unsigned char)c);
    }
    int i = 0, j = (int)t.size() - 1;
    bool ok = true;
    while (i < j) {
        if (t[i] != t[j]) { ok = false; break; }
        i++; j--;
    }
    cout << (ok ? "True" : "False") << endl;
}
