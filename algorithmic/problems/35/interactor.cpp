#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
double f(int x) {
    if (x <= 500) return 100;
    if (x > 5000) return 0;
    // Linear interpolation
    return 100.0 * (5000 - x) / (5000 - 500);
}

int a[605],t[305];
int main(int argc, char* argv[]) {
    registerInteraction(argc, argv);

    // cerr<<"ok\n";
    int T = 20;
    int nsum = 0;
    cout<<T<<endl;

    double score = 100.0;
    int mxq = 0;

    nsum = 0;
    // cerr<<"ok\n";
    for (int tc = 0; tc < T; tc++) {
        int n=300;
        mt19937 rnd(time(0));
        int pos=rnd()%n+1;
        int tot=0;
        for (int i = 1; i <= n; i++) {
            a[++tot] = i;
        }
        for(int i=1;i<pos;i++)a[++tot]=i;
        for(int i=pos+1;i<=n;i++)a[++tot]=i;
        assert(tot==2*n-1);
        shuffle(a+1,a+tot+1,rnd);
        cout<<n<<endl;
        // Find the position of n
        /*int pos = -1;
        for(int i=1;i<=n;i++)if(t[i]==1)pos=i;
        if (pos == -1) quitf(_fail, "value not found (test %d)", tc + 1);*/
        int queries = 0;
        while (true) {
            
            string op = ouf.readToken();
            if (op == "?") {
                int x = ouf.readInt(1, n);
                int m = ouf.readInt(1, n*2-1);
                bool flag=0;
                for(int i=1;i<=m;i++){
                    int nw=ouf.readInt(1,2*n-1);
                    if(a[nw]==x)flag=1;
                }
                cout<<flag<<endl;
                queries++;
                if(queries>5000)quitf(_wa,"Too many queries");

            } else if (op == "!") {
                int x = ouf.readInt(1, n, "x");
                if (x != pos) {
                    quitf(_wa, "wrong position on test %d: expected %d, got %d",
                          tc + 1, pos, x);
                }
                double sc = f(queries);
                score = min(score, sc);
                mxq=max(mxq,queries);
                break;
            } else {
                quitf(_pe, "unknown command '%s' on test %d", op.c_str(), tc + 1);
            }
        }
        // cerr<<queries<<"\n";
    }
    score/=100;
    quitp(score, "all %d tests passed; Ratio: %.4f; mxq = %d",
          T, score, mxq);
    return 0;
}
