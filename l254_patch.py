"""L254 -- cliff anatomy. WHAT fails when the frame is one rung below s_min?

L252 measured that the packer cannot exceed ~81.3% utilisation and that 83.7% of
the area deficit is that ceiling. It did not measure WHY. Two outcomes, opposite
verdicts:

  * failures concentrated in the LAST few small blocks  -> 81.3% is an artefact of
    an irrevocable greedy, and a cliff-edge backtrack could cross it;
  * failures are a LARGE block with nowhere to go       -> the ceiling is geometry
    and the whole line is closed by proof rather than by absence of a find.

Branches from the pristine shipping constructive.cpp and carries the L252
emitters too, so the ladder and the failure anatomy line up in one run.

    L254FAIL <frame_idx> <kind> <ndone> <N> <nblk> <iarea> <placed_area> <fw> <fh>

kind: PRE / PREOV  a preplaced block is outside the frame or overlaps -> the frame
                   is below the clamp, a structural miss, not a packing failure
      SINGLE       a free-aspect single block found no origin
      ITEM         an item (cluster or block) found no origin

Only the PRIMARY pack of each frame emits: ORDER_SWAP's hill-climb and REFINE's
guide passes both call pack_in_frame again, and their failures are not cliff
events. L254_PRIMARY gates that.
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l254.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

PATCHES = [
    # ---- flags -----------------------------------------------------------
    ('static bool FRAME_REPORT = false;',
     'static bool L252 = false;        // ICCAD_L252=1: emit the frame ladder\n'
     'static bool L254 = false;        // ICCAD_L254=1: emit pack failures\n'
     'static int  L254_FI = -1;        // frame index of the pack in flight\n'
     'static int  L254_PRIMARY = 0;    // 1 only during a frame\'s FIRST pack\n'
     'static bool FRAME_REPORT = false;'),
    ('if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;',
     'if (getenv("ICCAD_L252")) L252=true;\n'
     '    if (getenv("ICCAD_L254")) L254=true;\n'
     '    if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;'),
    # ---- the ladder (L252) -----------------------------------------------
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
    # ---- preplaced misses -------------------------------------------------
    (r'''        if (pos[i].x<-TOL||pos[i].y<-TOL||pos[i].x+pos[i].w>fw+TOL||pos[i].y+pos[i].h>fh+TOL) return false;
        for (const XYWH&r:rects) if (rect_overlap(pos[i].x,pos[i].y,pos[i].w,pos[i].h,r.x,r.y,r.w,r.h)) return false;''',
     r'''        if (pos[i].x<-TOL||pos[i].y<-TOL||pos[i].x+pos[i].w>fw+TOL||pos[i].y+pos[i].h>fh+TOL){
            if (L254&&L254_PRIMARY) fprintf(stderr,"L254FAIL %d PRE 0 %d 1 %.17g 0 %.17g %.17g\n",
                                            L254_FI,N,pos[i].w*pos[i].h,fw,fh);
            return false;
        }
        for (const XYWH&r:rects) if (rect_overlap(pos[i].x,pos[i].y,pos[i].w,pos[i].h,r.x,r.y,r.w,r.h)){
            if (L254&&L254_PRIMARY) fprintf(stderr,"L254FAIL %d PREOV 0 %d 1 %.17g 0 %.17g %.17g\n",
                                            L254_FI,N,pos[i].w*pos[i].h,fw,fh);
            return false;
        }'''),
    # ---- free-aspect single miss -----------------------------------------
    (r'''                if (!found) return false;
                out[sb]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1); bbox_add(bx,by,bw,bh); done[sb]=1;''',
     r'''                if (!found){
                    if (L254&&L254_PRIMARY){
                        double _pa=0; for (const XYWH&_r:rects) _pa+=_r.w*_r.h;
                        int _nd=0; for (int _i=0;_i<N;_i++) if (done[_i]) _nd++;
                        fprintf(stderr,"L254FAIL %d SINGLE %d %d 1 %.17g %.17g %.17g %.17g\n",
                                L254_FI,_nd,N,dims[sb].first*dims[sb].second,_pa,fw,fh);
                    }
                    return false;
                }
                out[sb]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1); bbox_add(bx,by,bw,bh); done[sb]=1;'''),
    # ---- item miss --------------------------------------------------------
    (r'''        if (!found) return false;
        for (size_t k=0;k<it.blocks.size();k++){''',
     r'''        if (!found){
            if (L254&&L254_PRIMARY){
                double _ia=0; for (int _b:it.blocks) _ia+=dims[_b].first*dims[_b].second;
                double _pa=0; for (const XYWH&_r:rects) _pa+=_r.w*_r.h;
                int _nd=0; for (int _i=0;_i<N;_i++) if (done[_i]) _nd++;
                fprintf(stderr,"L254FAIL %d ITEM %d %d %d %.17g %.17g %.17g %.17g\n",
                        L254_FI,_nd,N,(int)it.blocks.size(),_ia,_pa,fw,fh);
            }
            return false;
        }
        for (size_t k=0;k<it.blocks.size();k++){'''),
    # ---- the trial loop: index it, mark the primary pack, report both ways
    (r'''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frms){
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)) continue;''',
     r'''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    int l252_fi=-1;
    for (auto& f:frms){
        l252_fi++;
        items=items_base;
        vector<XYWH> c1, dummy;
        L254_FI=l252_fi; L254_PRIMARY=1;
        bool _ok0=run_frame(f.first,f.second,false,dummy,c1);
        L254_PRIMARY=0;
        if (!_ok0){
            if (L252) fprintf(stderr,"L252TRY %d 0 0\n",l252_fi);
            continue;
        }'''),
    (r'''        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true; }
        if (trials>=max_trials) break;''',
     r'''        if (L252) fprintf(stderr,"L252TRY %d 1 %.17g\n",l252_fi,sc);
        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true; }
        if (trials>=max_trials) break;'''),
]


def main():
    got = hashlib.md5(SRC.read_bytes()).hexdigest()
    if got != EXPECT_SRC_MD5:
        print("!! constructive.cpp is {} not {} -- the placer moved".format(
            got, EXPECT_SRC_MD5))
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
