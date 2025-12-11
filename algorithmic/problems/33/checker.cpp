#include "testlib.h"
using namespace std;

static string output_secret = "6cad0f33-b1bd-3a3e-1a8d-c4af23adfcbf";

const long long MX=1e18;
long long bit[5005];
int mx;

void init(int n)
{
	for(int i=0;i<=n+1;i++)
		bit[i]=0;
	mx=n+1;
}

long long add(long long x,long long y)
{
	x+=y;
	if(x>MX) return MX+1;
	return x;
}

void update(int ind,long long x)
{
	ind++;
	for(;ind<=mx;ind+=ind&(-ind))
		bit[ind]=add(bit[ind],x);
}

long long query(int ind)
{
	long long ret=0;
	ind++;
	for(;ind>0;ind-=ind&(-ind))
		ret=add(ret,bit[ind]);
	return ret;
}

long long getInc(vector<int> perm)
{
	int n=perm.size();
	init(n);
	long long ans=1;
	update(0,1);
	for(int i=0;i<n;i++)
	{
		long long me=query(perm[i]+1);
		update(perm[i]+1,me);
		ans=add(ans,me);
	}
	return ans;
}
double score(long long m) {
    double score;
    if (m <= 90) {
        score = 90.0;
    } else if (m <= 100) {
        score = 90.0 - 2.0 * (m - 90);   // 90 → 70
    } else if (m <= 120) {
        score = 70.0 - (m - 100);        // 69 → 50
    } else if (m <= 5000) {
        score = 50.0 - (m - 120) / 3.0;  // 49.67 → 0
    } else {
        score = 0.0;
    }
    if (score < 0) score = 0;
    return score;
}
int main(int argc, char * argv[])
{
	// registerChecker("perm", argc, argv);

	// readBothSecrets(output_secret);
	// readBothGraderResults();
	registerTestlibCmd(argc, argv);
    /*string output = ouf.readToken();
	if (output_secret != output)
		quitf(_wa, "output secret not match");
	string verdict = ouf.readToken();
	if (verdict != "OK")
		quitf(_wa, "verdict: %s", verdict.c_str());
	inf.readToken();*/
	int t=inf.readInt();
	int mxn=0;
	while(t--)
	{
		long long k=inf.readLong();
		int n=ouf.readInt(1,5000);
		vector<int> perm(n);
		set<int> ss;
		for(int i=0;i<n;i++)
		{
			perm[i]=ouf.readInt(0,n-1);
			ss.insert(perm[i]);
		}
		if(ss.size()!=n)
			quitf(_wa,"");
		long long g=getInc(perm);
		if(g!=k)
			quitf(_wa,"");
		mxn=max(mxn,n);
	}
	quitp(score(mxn),"Ratio: %.4f",score(mxn)/90.0);
}