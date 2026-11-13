#include <iostream>
#include <vector>
using namespace std;
int main(){
    int n; cin >> n;
    vector<long long> a(n), res;
    for (auto &x : a) cin >> x;
    for (auto x : a) {
        bool found = false;
        for (auto y : res) if (y == x) { found = true; break; }
        if (!found) res.push_back(x);
    }
    cout << res.size() << "\n";
    for (size_t i = 0; i < res.size(); i++) {
        cout << res[i];
        if (i + 1 < res.size()) cout << ' ';
    }
    cout << "\n";
}
