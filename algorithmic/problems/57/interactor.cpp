#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
double F(int x, int n) {
    int l=n,r=n+1000;
    if (x <= l) return 100;
    if (x > r) return 0;
    // Linear interpolation
    return 100.0 * (r - x) / (r - l);
}

int rt,a[1005],fa[1005],f[1005];
vector<int> g[1005];
void dfs(int x){
    f[x]+=a[x];
    for(auto i:g[x])if(i!=fa[x]){
        fa[i]=x,f[i]=f[x];dfs(i);
    }
}
void mdf(int x){
    f[x]-=a[x];
    a[x]=0-a[x];
    dfs(x);
}
int main(int argc, char* argv[]) {
    registerInteraction(argc, argv);

    // cerr<<"ok\n";
    int T = inf.readInt();
    int nsum = 0;
    cout<<T<<endl;

    double score = 100.0;
    int mxq = 0;

    nsum = 0;
    // cerr<<"ok\n";
    for (int tc = 0; tc < T; tc++) {
        int n=inf.readInt();
        
        cout<<n<<endl;
        mt19937 rnd(time(0));
        for(int i=1;i<=n;i++){
            g[i].clear();
            a[i]=rnd()%2;
            if(a[i]==0)a[i]=-1;
            fa[i]=0,f[i]=0;
        }   
        for(int i=1;i<n;i++){
            int x=inf.readInt(1,n),y=inf.readInt(1,n);
            assert(x!=y);
            cout<<x<<" "<<y<<endl;
            g[x].push_back(y),g[y].push_back(x);
        }
        rt=inf.readInt(1,n);
        dfs(rt);
        // cerr<<tc<<" ok\n";
        int queries = 0;
        while (true) {
            
            string op = ouf.readToken();
            if (op == "?") {
                queries++;
                int op1=ouf.readInt();
                if(op1==1){
                    int m=ouf.readInt(1,n);
                    int res=0;
                    for(int i=1;i<=m;i++){
                        int x=ouf.readInt(1,n);
                        res+=f[x];
                    }
                    cout<<res<<endl;
                }
                else if(op1==2){
                    int x=ouf.readInt(1,n);
                    mdf(x);
                }
                else quitf(_wa,"unknown command");
            } else if (op == "!") {
                bool flag=1;
                for(int i=1;i<=n;i++){
                    int x=ouf.readInt();
                    if(x!=a[i]){
                        quitf(_wa,"wrong answer");
                    }
                }
                double sc = F(queries,n);
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
