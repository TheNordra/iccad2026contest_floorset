"""L252 - instrument frame_candidates() so the frame ladder is observable.

Branched from the CURRENT shipping constructive.cpp (md5 e2c7b2f4...), the same
discipline L249 used: a probe binary anchored to a placer that has since moved
is the ledger's recurring failure mode.

Adds three stderr emitters, ALL gated behind ICCAD_L252=1 so the default path is
bit-identical to the shipped binary (proved by l252_identity.py, not asserted):

    L252TOT <sum of block areas>        once, from frame_candidates()
    L252FRM <i> <w> <h>                 the full candidate ladder, post-sort
    L252TRY <i> <ok> <layout_score>     per trial; ok=0 means the frame did not pack

From these:  s_i = sqrt(w_i*h_i / TOT)  and  utilisation_i = 1/s_i^2
             s_min    = min s_i with ok=1        (the cliff, on OUR packer)
             s_landed = the argmin of score over ok=1   (what layout_score picks)
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l252.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

PATCHES = [
    # 1. the flag
    ('static bool FRAME_REPORT = false;',
     'static bool L252 = false;        // ICCAD_L252=1: emit the frame ladder\n'
     'static bool FRAME_REPORT = false;'),
    # 2. env parse, next to the other L108 knobs
    ('if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;',
     'if (getenv("ICCAD_L252")) L252=true;\n'
     '    if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;'),
    # 3. dump the ladder itself (post-sort, so indices match the trial loop)
    ('''    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    return frames;''',
     '''    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    if (L252){
        fprintf(stderr,"L252TOT %.17g\\n",total);
        for (size_t i=0;i<frames.size();i++)
            fprintf(stderr,"L252FRM %d %.17g %.17g\\n",(int)i,frames[i].first,frames[i].second);
    }
    return frames;'''),
    # 4. the trial loop: index it, and report BOTH outcomes
    ('''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frms){
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)) continue;''',
     '''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    int l252_fi=-1;
    for (auto& f:frms){
        l252_fi++;
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)){
            if (L252) fprintf(stderr,"L252TRY %d 0 0\\n",l252_fi);
            continue;
        }'''),
    # 5. the score the sequential loop actually compares on (post hill-climb,
    #    post REFINE) -- the same quantity FSEL reports for the winner
    ('''        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true; }
        if (trials>=max_trials) break;''',
     '''        if (L252) fprintf(stderr,"L252TRY %d 1 %.17g\\n",l252_fi,sc);
        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true; }
        if (trials>=max_trials) break;'''),
]


def main():
    src = SRC.read_bytes().decode("utf-8")
    got = hashlib.md5(SRC.read_bytes()).hexdigest()
    if got != EXPECT_SRC_MD5:
        print("!! constructive.cpp is {} not {} -- the placer moved under this"
              " patch. Re-derive before trusting anything.".format(got, EXPECT_SRC_MD5))
        return 1
    out = src
    for i, (old, new) in enumerate(PATCHES, 1):
        if out.count(old) != 1:
            print("!! patch {} matches {} times, expected exactly 1"
                  .format(i, out.count(old)))
            return 1
        out = out.replace(old, new)
    DST.write_bytes(out.encode("utf-8"))
    print("wrote {}  ({} patches, {} -> {} bytes)"
          .format(DST.name, len(PATCHES), len(src), len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
