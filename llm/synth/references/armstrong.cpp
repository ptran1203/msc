#include <iostream>
using namespace std;
int main(){
    long long n;
    cin >> n;
    long long temp = n;
    int d = 0;
    while (temp > 0) { d++; temp /= 10; }
    long long sum = 0;
    temp = n;
    while (temp > 0) {
        int digit = temp % 10;
        long long p = 1;
        for (int i = 0; i < d; i++) p *= digit;
        sum += p;
        temp /= 10;
    }
    cout << (sum == n ? "True" : "False") << endl;
}
