#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include "testlib.h"
using namespace std;
enum {
    INVALID_INPUT = -1,        // Judge input invalid (internal use only)
    INVALID_T_LENGTH = 1,      // t length != N
    INVALID_T_CHAR   = 2,      // t contains illegal characters
    WRONG_T          = 3,      // t does not match true value
    INVALID_S_LENGTH = 4,      // s length != 2N+1
    INVALID_S_CHAR   = 5,      // s contains illegal characters
    QUERY_LIMIT_EXCEEDED = 6,  // query count > 1000
    INVALID_OUTPUT   = 9,      // player output first char is neither '?' nor '!'
};

const int N_MAX = 8000;
const int QUERY_LIMIT = 5000;

int N, R;
std::vector<int> U, V;
std::string T;
int QUERY_COUNT = 0;

double score(int x){
    if(x<=900)return 1;
    if(x>5000)return 0;
    return (5000.0-x)/(5000.0-900.0);
}

[[noreturn]] void wrong(const int num) {
    // Write a flag to player program (avoid blocking read), then terminate via testlib
    fprintf(stdout, "-1\n");
    fflush(stdout);
    quitf(_wa, "translate:wrong\nWrong Answer [%d]\n", num);
}

[[noreturn]] void ok() {
    quitp(score(QUERY_COUNT), "Ratio: %.4f , Queries: %d", score(QUERY_COUNT), QUERY_COUNT);
}

int query(std::string s) {
    const int M = 2 * N + 1;
    if ((int)s.size() != M) {
        wrong(INVALID_S_LENGTH);
    }
    for (char c : s) {
        if (c != '0' && c != '1') wrong(INVALID_S_CHAR);
    }
    if (QUERY_COUNT == QUERY_LIMIT) {
        wrong(QUERY_LIMIT_EXCEEDED);
    }
    QUERY_COUNT++;

    // Convert '0'/'1' to 0/1
    for (char &c : s) c -= '0';

    // Calculate slot outputs from high to low and fold to switch i (XOR trick for OFF/ON behaviors)
    for (int i = N - 1; i >= 0; --i) {
        const int u = U[i], v = V[i];
        if (T[i] == '&') {
            s[i] ^= (s[u] & s[v]);
        } else { // '|'
            s[i] ^= (s[u] | s[v]);
        }
    }
    // Return final output of switch 0
    return s[0];
}

void answer(std::string t) {
    if ((int)t.size() != N) {
        wrong(INVALID_T_LENGTH);
    }
    for (char c : t) {
        if (c != '&' && c != '|') wrong(INVALID_T_CHAR);
    }
    if (t != T) {
        wrong(WRONG_T);
    }
    ok();
}

int main(int argc, char* argv[]) {
    registerInteraction(argc, argv);

    // ---------- Read judge input (internal use only) ----------
    N = inf.readInt();     // 1..8000
    R = inf.readInt();     // 1..min(N,120)
    if (N < 1 || N > N_MAX) {
        wrong(INVALID_INPUT);
    }
    cout<<N<<" "<<R<<endl;
    U.resize(N);
    V.resize(N);
    for (int i = 0; i < N; ++i) {
        U[i] = inf.readInt();
        V[i] = inf.readInt();
        /*if (!(i < U[i] && U[i] < V[i] && V[i] <= 2 * N)) {
            wrong(INVALID_INPUT);
        }*/
       cout<<U[i]<<" "<<V[i]<<endl;
    }

    // Hidden truth value (internal use only, not visible to players)
    T = inf.readToken();
    if ((int)T.size() != N) {
        wrong(INVALID_INPUT);
    }
    int orCount = 0;
    for (char c : T) {
        if (c != '&' && c != '|') wrong(INVALID_INPUT);
        if (c == '|') ++orCount;
    }
    if (orCount > R) {
        // Judge data inconsistent with declared R
        wrong(INVALID_INPUT);
    }

    // ---------- Output initial visible info to player ----------
    // Format: N R, then N lines of U[i] V[i]
    /*fprintf(stdout, "%d %d\n", N, R);
    for (int i = 0; i < N; ++i) {
        fprintf(stdout, "%d %d\n", U[i], V[i]);
    }
    fflush(stdout);*/

    // ---------- Interaction loop ----------
    while (true) {
        // Read operator ('?', '!')
        std::string op = ouf.readToken();
        if (op.empty()) wrong(INVALID_OUTPUT);
        const char type = op[0];
        if (type != '?' && type != '!') wrong(INVALID_OUTPUT);

        // Read following string (s or t)
        std::string payload = ouf.readToken();

        // Compatibility: remove trailing newline if present (readToken usually doesn't include it)
        if (!payload.empty() && payload.back() == '\n') payload.pop_back();

        if (type == '?') {
            // Handle query
            int res = query(payload);
            fprintf(stdout, "%d\n", res);
            fflush(stdout);
        } else {
            // Final answer
            answer(payload);
            // answer() will call quitf internally; should not reach here normally
        }
    }
}
