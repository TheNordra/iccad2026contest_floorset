"""L259 -- dump the JAM STATE so the recreate sub-problem can be solved exactly.

L256 ruined ~12% of the design and re-placed it with the SAME greedy that had just
jammed; that is why it only went ~2% deep. L253 says the topology is already right
and L254 says the jam is fragmentation, so the missing piece is a RECREATE that
can place into fragmented space -- a small combinatorial problem (median 3 items,
per L254) that nothing in the ledger has attacked.

Before building one, bound it: at the moment the greedy gives up, is there ANY
legal placement of the remaining blocks in the free space? That is decidable by
bounded backtracking IF we can see the state, which is what this emits.

    L259JAM <frame_idx> <fw> <fh> <ndone> <N>
    L259P   <i> <x> <y> <w> <h>     every block already placed
    L259U   <i> <w> <h>             every block still unplaced (nominal dims)

Gated behind ICCAD_L259, primary pack only (ORDER_SWAP's hill-climb and REFINE's
guide passes call pack_in_frame again and their failures are not cliff events).
Carries the L252 ladder emitters so the frame scale is known.
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l259.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

DUMP = r'''
#define L259_DUMP(TAG) do { if (L259 && L259_PRIMARY) { \
    fprintf(stderr,"L259JAM %d %.17g %.17g %d %d\n",L259_FI,fw,fh,_nd259,N); \
    for (int _i=0;_i<N;_i++){ \
        if (done[_i]) fprintf(stderr,"L259P %d %.17g %.17g %.17g %.17g\n", \
                              _i,out[_i].x,out[_i].y,out[_i].w,out[_i].h); \
        else fprintf(stderr,"L259U %d %.17g %.17g\n",_i,dims[_i].first,dims[_i].second); \
    } } } while(0)
'''

PATCHES = [
    ('static bool FRAME_REPORT = false;',
     'static bool L252 = false;        // ICCAD_L252=1: emit the frame ladder\n'
     'static bool L259 = false;        // ICCAD_L259=1: dump the jam state\n'
     'static int  L259_FI = -1;\n'
     'static int  L259_PRIMARY = 0;\n'
     'static bool FRAME_REPORT = false;'),
    ('if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;',
     'if (getenv("ICCAD_L252")) L252=true;\n'
     '    if (getenv("ICCAD_L259")) L259=true;\n'
     '    if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;'),
    # the ladder, so the frame scale of each jam is known
    (r'''    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    return frames;''',
     r'''    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    if (L252){
        fprintf(stderr,"L252TOT %.17g\n",total);
        for (size_t i=0;i<frames.size();i++)
            fprintf(stderr,"L252FRM %d %.17g %.17g\n",(int)i,frames[i].first,frames[i].second);
    }
    return frames;'''),
    # free-aspect single miss
    (r'''                if (!found) return false;
                out[sb]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1); bbox_add(bx,by,bw,bh); done[sb]=1;''',
     r'''                if (!found){
                    if (L259 && L259_PRIMARY){
                        int _nd259=0; for (int _i=0;_i<N;_i++) if (done[_i]) _nd259++;
                        fprintf(stderr,"L259JAM %d %.17g %.17g %d %d\n",L259_FI,fw,fh,_nd259,N);
                        for (int _i=0;_i<N;_i++){
                            if (done[_i]) fprintf(stderr,"L259P %d %.17g %.17g %.17g %.17g\n",
                                                  _i,out[_i].x,out[_i].y,out[_i].w,out[_i].h);
                            else fprintf(stderr,"L259U %d %.17g %.17g\n",_i,dims[_i].first,dims[_i].second);
                        }
                    }
                    return false;
                }
                out[sb]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1); bbox_add(bx,by,bw,bh); done[sb]=1;'''),
    # item miss
    (r'''        if (!found) return false;
        for (size_t k=0;k<it.blocks.size();k++){''',
     r'''        if (!found){
            if (L259 && L259_PRIMARY){
                int _nd259=0; for (int _i=0;_i<N;_i++) if (done[_i]) _nd259++;
                fprintf(stderr,"L259JAM %d %.17g %.17g %d %d\n",L259_FI,fw,fh,_nd259,N);
                for (int _i=0;_i<N;_i++){
                    if (done[_i]) fprintf(stderr,"L259P %d %.17g %.17g %.17g %.17g\n",
                                          _i,out[_i].x,out[_i].y,out[_i].w,out[_i].h);
                    else fprintf(stderr,"L259U %d %.17g %.17g\n",_i,dims[_i].first,dims[_i].second);
                }
            }
            return false;
        }
        for (size_t k=0;k<it.blocks.size();k++){'''),
    # index the trial loop and mark the primary pack
    (r'''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frms){
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)) continue;''',
     r'''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    int l259_fi=-1;
    for (auto& f:frms){
        l259_fi++;
        items=items_base;
        vector<XYWH> c1, dummy;
        L259_FI=l259_fi; L259_PRIMARY=1;
        bool _ok0=run_frame(f.first,f.second,false,dummy,c1);
        L259_PRIMARY=0;
        if (!_ok0){
            if (L252) fprintf(stderr,"L252TRY %d 0 0\n",l259_fi);
            continue;
        }'''),
    (r'''        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true; }
        if (trials>=max_trials) break;''',
     r'''        if (L252) fprintf(stderr,"L252TRY %d 1 %.17g\n",l259_fi,sc);
        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true; }
        if (trials>=max_trials) break;'''),
]


def main():
    got = hashlib.md5(SRC.read_bytes()).hexdigest()
    if got != EXPECT_SRC_MD5:
        print("!! constructive.cpp is {} not {}".format(got, EXPECT_SRC_MD5))
        return 1
    out = SRC.read_bytes().decode("utf-8")
    for i, (old, new) in enumerate(PATCHES, 1):
        if out.count(old) != 1:
            print("!! patch {} matches {} times, expected 1".format(i, out.count(old)))
            return 1
        out = out.replace(old, new)
    DST.write_bytes(out.encode("utf-8"))
    print("wrote {}  ({} patches)".format(DST.name, len(PATCHES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
