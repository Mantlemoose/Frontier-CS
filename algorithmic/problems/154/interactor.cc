#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static const int D = 30;
static const int T = 300;

struct Input {
    vector<tuple<int,int,int>> ps; // pets: x,y,t (0-based)
    vector<pair<int,int>> hs;      // humans: x,y (0-based)
    unsigned long long seed;
};

struct Sim {
    vector<vector<bool>> blocked;
    mt19937_64 rng;
    struct PetState {
        int x, y;
        int type; // 1..5
        int target_h; // for dog: human index or -1
        int tx, ty;   // for cat: target coords or -1
    };
    vector<PetState> ps;
    vector<pair<int,int>> hs;
    int turn;

    Sim(const Input& in): blocked(D, vector<bool>(D, false)), rng(in.seed), hs(in.hs), turn(0) {
        ps.reserve(in.ps.size());
        for (auto &t : in.ps) {
            int x,y,tp; tie(x,y,tp)=t;
            PetState s; s.x=x; s.y=y; s.type=tp; s.target_h=-1; s.tx=-1; s.ty=-1;
            ps.push_back(s);
        }
    }

    static inline bool inside(int x, int y) {
        return 0 <= x && x < D && 0 <= y && y < D;
    }

    vector<vector<int>> bfs(int sx, int sy) const {
        const int INF = INT_MAX/2;
        vector<vector<int>> dist(D, vector<int>(D, INF));
        queue<pair<int,int>> q;
        dist[sx][sy] = 0;
        q.push({sx,sy});
        static const int dx[4]={-1,1,0,0};
        static const int dy[4]={0,0,-1,1};
        while(!q.empty()){
            auto [x,y]=q.front(); q.pop();
            int t = dist[x][y];
            for(int d=0; d<4; d++){
                int x2=x+dx[d], y2=y+dy[d];
                if(inside(x2,y2) && !blocked[x2][y2] && dist[x2][y2] > t+1){
                    dist[x2][y2]=t+1;
                    q.push({x2,y2});
                }
            }
        }
        return dist;
    }

    int standard_move(int &x, int &y) {
        static const int dx[4]={-1,1,0,0};
        static const int dy[4]={0,0,-1,1};
        vector<int> cand;
        for(int d=0; d<4; d++){
            int x2=x+dx[d], y2=y+dy[d];
            if(inside(x2,y2) && !blocked[x2][y2]) cand.push_back(d);
        }
        // From problem constraints, such squares always exist.
        int d = cand[uniform_int_distribution<int>(0,(int)cand.size()-1)(rng)];
        x += dx[d];
        y += dy[d];
        return d;
    }

    string pet_move() {
        static const char DIR_CHAR[5] = {'U','D','L','R','.'};
        static const int dx[4]={-1,1,0,0};
        static const int dy[4]={0,0,-1,1};
        auto ps_copy = ps;
        vector<string> tokens;
        tokens.reserve(ps_copy.size());
        for (auto &p : ps_copy) {
            if (p.type == 1) { // Cow
                int d = standard_move(p.x, p.y);
                tokens.emplace_back(string(1, DIR_CHAR[d]));
            } else if (p.type == 2) { // Pig
                string s;
                for(int i=0;i<2;i++){
                    int d = standard_move(p.x, p.y);
                    s.push_back(DIR_CHAR[d]);
                }
                tokens.emplace_back(s);
            } else if (p.type == 3) { // Rabbit
                string s;
                for(int i=0;i<3;i++){
                    int d = standard_move(p.x, p.y);
                    s.push_back(DIR_CHAR[d]);
                }
                tokens.emplace_back(s);
            } else if (p.type == 4) { // Dog
                while(true){
                    if (p.target_h == -1 || (hs[p.target_h].first == p.x && hs[p.target_h].second == p.y)) {
                        auto dist = bfs(p.x, p.y);
                        vector<int> cand;
                        for(int i=0;i<(int)hs.size();i++){
                            int tx=hs[i].first, ty=hs[i].second;
                            if (dist[tx][ty] != INT_MAX/2 && dist[tx][ty] > 0) cand.push_back(i);
                        }
                        if (!cand.empty()) {
                            p.target_h = cand[uniform_int_distribution<int>(0,(int)cand.size()-1)(rng)];
                        } else {
                            p.target_h = -1;
                            int d = standard_move(p.x, p.y);
                            tokens.emplace_back(string(1, DIR_CHAR[d]));
                            break;
                        }
                    }
                    int tx = hs[p.target_h].first, ty = hs[p.target_h].second;
                    auto dist = bfs(tx, ty);
                    if (dist[p.x][p.y] == INT_MAX/2) {
                        p.target_h = -1;
                        continue;
                    }
                    vector<int> candd;
                    for(int d=0; d<4; d++){
                        int x2=p.x+dx[d], y2=p.y+dy[d];
                        if (inside(x2,y2) && dist[p.x][p.y] > dist[x2][y2]) candd.push_back(d);
                    }
                    int dir = candd[uniform_int_distribution<int>(0,(int)candd.size()-1)(rng)];
                    p.x += dx[dir]; p.y += dy[dir];
                    if (hs[p.target_h].first == p.x && hs[p.target_h].second == p.y) {
                        p.target_h = -1;
                    }
                    string s;
                    s.push_back(DIR_CHAR[dir]);
                    int d2 = standard_move(p.x, p.y);
                    s.push_back(DIR_CHAR[d2]);
                    if (p.target_h != -1 && hs[p.target_h].first == p.x && hs[p.target_h].second == p.y) {
                        p.target_h = -1;
                    }
                    tokens.emplace_back(s);
                    break;
                }
            } else if (p.type == 5) { // Cat
                while(true){
                    if (p.tx != -1 && blocked[p.tx][p.ty]) {
                        p.tx = p.ty = -1;
                    }
                    if (p.tx == -1) {
                        auto dist = bfs(p.x, p.y);
                        vector<pair<int,int>> cand;
                        for(int i=0;i<D;i++){
                            for(int j=0;j<D;j++){
                                if (dist[i][j] != INT_MAX/2 && dist[i][j] > 0) cand.emplace_back(i,j);
                            }
                        }
                        // In original tools, they assume exists; choose one
                        auto pr = cand[uniform_int_distribution<int>(0,(int)cand.size()-1)(rng)];
                        p.tx = pr.first; p.ty = pr.second;
                    }
                    auto dist = bfs(p.tx, p.ty);
                    if (dist[p.x][p.y] == INT_MAX/2) {
                        p.tx = p.ty = -1;
                        continue;
                    }
                    vector<int> candd;
                    for(int d=0; d<4; d++){
                        int x2=p.x+dx[d], y2=p.y+dy[d];
                        if (inside(x2,y2) && dist[p.x][p.y] > dist[x2][y2]) candd.push_back(d);
                    }
                    int dir = candd[uniform_int_distribution<int>(0,(int)candd.size()-1)(rng)];
                    p.x += dx[dir]; p.y += dy[dir];
                    if (p.tx == p.x && p.ty == p.y) { p.tx = p.ty = -1; }
                    string s;
                    s.push_back(DIR_CHAR[dir]);
                    int d2 = standard_move(p.x, p.y);
                    s.push_back(DIR_CHAR[d2]);
                    if (p.tx == p.x && p.ty == p.y) { p.tx = p.ty = -1; }
                    tokens.emplace_back(s);
                    break;
                }
            }
        }
        // commit
        ps = ps_copy;
        // join tokens by space
        string ret;
        for (int i = 0; i < (int)tokens.size(); i++) {
            if (i) ret.push_back(' ');
            ret += tokens[i];
        }
        return ret;
    }

    void human_move(const string &out_raw) {
        turn += 1;
        // trim and validate length
        string out = out_raw;
        auto trim_str = [](const string &s)->string{
            size_t l=0, r=s.size();
            while(l<r && isspace((unsigned char)s[l])) l++;
            while(r>l && isspace((unsigned char)s[r-1])) r--;
            return s.substr(l, r-l);
        };
        string s = trim_str(out);
        if ((int)s.size() != (int)hs.size()) {
            quitf(_wa, "illegal output length (turn %d)", turn);
        }
        vector<pair<int,int>> new_hs = hs;
        // Apply blockings
        for (int i = 0; i < (int)hs.size(); i++) {
            char c = s[i];
            if (c == '.') continue;
            if (islower((unsigned char)c)) {
                int dir = -1;
                if (c=='u') dir=0; else if (c=='d') dir=1; else if (c=='l') dir=2; else if (c=='r') dir=3;
                if (dir==-1) quitf(_wa, "illegal output char %c (turn %d)", c, turn);
                static const int dx[4]={-1,1,0,0};
                static const int dy[4]={0,0,-1,1};
                int x = hs[i].first + dx[dir];
                int y = hs[i].second + dy[dir];
                if (!inside(x,y)) continue;
                // cannot block square containing pet
                for (auto &p : ps) {
                    if (p.x==x && p.y==y) {
                        quitf(_wa, "trying to block a square containing a pet (turn %d)", turn);
                    }
                }
                // cannot block square containing human
                for (auto &h : hs) {
                    if (h.first==x && h.second==y) {
                        quitf(_wa, "trying to block a square containing a human (turn %d)", turn);
                    }
                }
                // cannot block square whose adjacent contains a pet
                for (int d2=0; d2<4; d2++){
                    int x2=x + dx[d2], y2=y + dy[d2];
                    if (inside(x2,y2)) {
                        for (auto &p : ps) {
                            if (p.x==x2 && p.y==y2) {
                                quitf(_wa, "trying to block a square whose adjacent square contains a pet (turn %d)", turn);
                            }
                        }
                    }
                }
                blocked[x][y] = true; // if already blocked, nothing happens
            } else if (isupper((unsigned char)c)) {
                int dir = -1;
                if (c=='U') dir=0; else if (c=='D') dir=1; else if (c=='L') dir=2; else if (c=='R') dir=3;
                if (dir==-1) quitf(_wa, "illegal output char %c (turn %d)", c, turn);
                static const int dx[4]={-1,1,0,0};
                static const int dy[4]={0,0,-1,1};
                new_hs[i].first += dx[dir];
                new_hs[i].second += dy[dir];
                if (!inside(new_hs[i].first, new_hs[i].second)) {
                    quitf(_wa, "trying to move to an impassible square (turn %d)", turn);
                }
            } else {
                quitf(_wa, "illegal output char %c (turn %d)", c, turn);
            }
        }
        hs = new_hs;
        for (int i = 0; i < (int)hs.size(); i++) {
            if (blocked[hs[i].first][hs[i].second]) {
                quitf(_wa, "trying to move to an impassible square (turn %d)", turn);
            }
        }
    }

    long long compute_score() const {
        double score = 0.0;
        for (auto &h : hs) {
            auto dist = bfs(h.first, h.second);
            int r = 0;
            for (int i=0;i<D;i++) for(int j=0;j<D;j++) if (dist[i][j] != INT_MAX/2) r++;
            int n = 0;
            for (auto &p : ps) if (dist[p.x][p.y] != INT_MAX/2) n++;
            score += (double)r / (30.0 * 30.0) * pow(2.0, -n);
        }
        long long scaled = llround(1e8 * score / (double)hs.size());
        return scaled;
    }
};

static string trim_str_all(const string &s) {
    size_t l=0, r=s.size();
    while (l<r && isspace((unsigned char)s[l])) l++;
    while (r>l && isspace((unsigned char)s[r-1])) r--;
    return s.substr(l, r-l);
}

int main(int argc, char* argv[]) {
    registerInteraction(argc, argv);

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Read input from inf
    Input in;
    int Npets = inf.readInt();
    in.ps.resize(Npets);
    for (int i=0;i<Npets;i++){
        int x = inf.readInt();
        int y = inf.readInt();
        int t = inf.readInt();
        // convert to 0-based
        get<0>(in.ps[i]) = x-1;
        get<1>(in.ps[i]) = y-1;
        get<2>(in.ps[i]) = t;
    }
    int Mhum = inf.readInt();
    in.hs.resize(Mhum);
    for (int i=0;i<Mhum;i++){
        int x = inf.readInt();
        int y = inf.readInt();
        in.hs[i] = {x-1, y-1};
    }
    in.seed = inf.readUnsignedLong();

    // Send initial positions (without seed) to participant
    cout << Npets << "\n";
    for (int i=0;i<Npets;i++){
        cout << get<0>(in.ps[i]) + 1 << " " << get<1>(in.ps[i]) + 1 << " " << get<2>(in.ps[i]) << "\n";
    }
    cout << Mhum << "\n";
    for (int i=0;i<Mhum;i++){
        cout << in.hs[i].first + 1 << " " << in.hs[i].second + 1 << "\n";
    }
    cout.flush();

    Sim sim(in);

    for (int t=1; t<=T; t++){
        // Read a non-empty, non-comment line from participant
        string line;
        do {
            line = ouf.readLine();
            line = trim_str_all(line);
        } while (line.empty() || (!line.empty() && line[0]=='#'));

        sim.human_move(line);

        string pet_mv = sim.pet_move();
        cout << pet_mv << "\n";
        cout.flush();
    }

    long long score = sim.compute_score();
    // long long baseline_value = ans.readLong();
    // long long best_value = ans.readLong();

    // double score_ratio = max(0.0, min(1.0, (double)(score - baseline_value) / (best_value - baseline_value)));
    // quitp(score_ratio, "Value: %lld. Ratio: %.4f", score, score_ratio);
    quitf(_wa, "score (%lld)", score);
    return 0;
}