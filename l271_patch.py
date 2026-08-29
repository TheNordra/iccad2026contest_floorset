"""L271 -- buy the density WITHOUT paying the wire.

THE FINDING THIS ATTACKS (L267_L269_REPORT 2.2 / 2.3 / 2.5).
`ICCAD_L268=4` ("nosize") drops exactly one tie-break -- compound cluster items no
longer go before larger singles -- and moves the packer's density ceiling from
81.34 % to 84.83 % utilisation, area_gap 0.2300 -> 0.1972, with violations BETTER
than shipped. It still loses, because it pays hpwl 0.2924 -> 0.3332. Cost prices
hpwl_gap and area_gap identically, so the exchange rate is 0.0408/0.0328 = 1.24
and the mechanism is 24 % the wrong side of a line that sits exactly at 1.0.

WHY THE WIRE COLLAPSES, IN ONE LINE OF SOURCE.  constructive.cpp:1128

    if (done[nb.first]){ ... }
    else if (use_prev){ ... }
    else continue;                 <-- an unplaced neighbour contributes NOTHING

The greedy's wire term only sees neighbours that are ALREADY PLACED. Compound
cluster items are the connectivity anchors, so demoting them makes every early
placement wire-blind. The reordering does not damage the wire directly; it
destroys the *information* the wire term runs on.

THE REPAIR IS ALREADY IN THE BINARY, POINTED THE WRONG WAY.  `use_prev` /
`prev_pos` make the wire term also pull toward NOT-yet-placed neighbours at their
guide positions -- that is what REFINE does. But the guide is always the same
order's own previous layout, so under `nosize` REFINE starts from, and converges
around, the wire-scrambled pack it is supposed to fix.

WHAT L271 DOES.  The frames that fail before the shipped order's first success are
exactly the density prize: they are tighter than anything the shipped order can
reach, and `nosize` can reach them. So when the shipped order finally succeeds:

  1. take the tightest frame that just FAILED,
  2. scale the successful layout into it (frame-relative, like the L137 hint),
  3. re-pack that tight frame in the DENSITY order with that layout as the guide.

The density order now has full wire visibility from its very first pass -- from a
layout produced by the wire-sane order, on the same instance. Nothing is packed
speculatively: the failing packs were already paid for by the shipped walk.

    ICCAD_L252=1      stderr instrumentation only (ladder, per-trial ok, L271HIT)
    ICCAD_L268=4      the "nosize" density order on its own (reproduces the
                      other session's arm -- a cross-check, not a new mechanism)
    ICCAD_L271=1      retry tightest-K failed frames, DENSITY order, WITH guide
    ICCAD_L271=2      ... SHIPPED order, WITH guide      (isolates the order)
    ICCAD_L271=4      ... DENSITY order, NO guide        (isolates the guide)
    ICCAD_L271=5      retry the SAME frame in the DENSITY order WITH the guide --
                      antecedent is never empty, and area_gap is the layout bbox
                      rather than the frame, so a denser pack still buys area
    ICCAD_L271=6      ... SAME frame, DENSITY order, NO guide  (control for 5)
    ICCAD_L271_K=N    how many failed frames to retry    (default 1)
    ICCAD_L271_REF=0  do not give the retry its own REFINE passes (default: do,
                      so the retry is compared fairly against a refined incumbent)
    ICCAD_L272=1      the L137 GORDIAN hint feeds the WIRE term for unplaced
                      neighbours -- order-independent wire visibility, zero packs

All flags default off => bit-identical to the shipped binary.
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l271.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

PATCHES = [
    # 1 flags
    ('static bool FRAME_REPORT = false;',
     'static bool L252 = false;        // ICCAD_L252=1: stderr instrumentation only\n'
     'static int  L268 = 0;            // ICCAD_L268=4: the "nosize" density order\n'
     'static int  L271 = 0;            // ICCAD_L271: 1 dens+guide, 2 ship+guide, 4 dens only\n'
     'static int  L271_K = 1;          // ICCAD_L271_K: failed frames retried\n'
     'static int  L271_REF = 1;        // ICCAD_L271_REF=0: no REFINE on the retry\n'
     'static long L271_PACKS = 0;      // pack_in_frame calls -- attempts, NOT wall\n'
     'static bool L272 = false;        // ICCAD_L272=1: the L137 hint feeds the WIRE term\n'
     'static bool FRAME_REPORT = false;'),
    # 2 env
    ('    if (getenv("ICCAD_GUIDE_MED")) GUIDE_MED=true;',
     '    if (getenv("ICCAD_L252")) L252=true;\n'
     '    if (const char* e=getenv("ICCAD_L268")){ int v=atoi(e); if (v==4) L268=v; }\n'
     '    if (const char* e=getenv("ICCAD_L271")){ int v=atoi(e); if (v>=1&&v<=6) L271=v; }\n'
     '    if (const char* e=getenv("ICCAD_L271_K")){ int v=atoi(e); if (v>0) L271_K=v; }\n'
     '    if (const char* e=getenv("ICCAD_L271_REF")){ L271_REF=atoi(e); }\n'
     '    if (getenv("ICCAD_L272")) L272=true;\n'
     '    if (getenv("ICCAD_GUIDE_MED")) GUIDE_MED=true;'),
    # 3 pack counter. 🚨 NOT a wall proxy -- L267 measured 1.063x the packs against
    #   1.2417x the wall, because a max over pack COUNT picks the profile that does
    #   many cheap packs, not the one that sets the time. Attempts, not seconds.
    ('static bool pack_in_frame(double fw,double fh,const vector<Item>& items,vector<XYWH>& out){\n'
     '    out=pos; vector<XYWH> rects; bbox_reset();',
     'static bool pack_in_frame(double fw,double fh,const vector<Item>& items,vector<XYWH>& out){\n'
     '    L271_PACKS++;\n'
     '    out=pos; vector<XYWH> rects; bbox_reset();'),
    # 4 the L252 ladder emitters
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
    # 5 the density order, standalone
    ('''    vector<int> order; for(int i=0;i<N;i++) if(!blocks[i].is_preplaced) order.push_back(i);''',
     '''    // L268=4 "nosize": keep the bscore boundary CLASS intact -- that key is what
    // holds violations down, and dropping it costs vrel 0.0893 -> 0.0998 -- and
    // remove only "compound cluster items before singles", so a large single stops
    // waiting behind small compound items.
    if (L268==4){
        stable_sort(items.begin(),items.end(),[](const Item&a,const Item&b){
            if (a.bscore!=b.bscore) return a.bscore>b.bscore;
            return a.w*a.h > b.w*b.h;
        });
    }

    vector<int> order; for(int i=0;i<N;i++) if(!blocks[i].is_preplaced) order.push_back(i);'''),
    # 6 keep both orders live
    ('    vector<Item> items_base=items;   // original sorted order, captured once (reframe reuses it)',
     '    vector<Item> items_base=items;   // original sorted order, captured once (reframe reuses it)\n'
     '    // L271 needs BOTH orders at once: the shipped one (wire-sane, because the\n'
     '    // compound cluster items that lead it ARE the connectivity anchors) and the\n'
     '    // density one. Sorting is free; only packing is expensive.\n'
     '    vector<Item> items_dens=items;\n'
     '    if (L271) stable_sort(items_dens.begin(),items_dens.end(),\n'
     '        [](const Item&a,const Item&b){\n'
     '            if (a.bscore!=b.bscore) return a.bscore>b.bscore;\n'
     '            return a.w*a.h > b.w*b.h;\n'
     '        });'),
    # 7 the retry
    ('''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frms){
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)) continue;''',
     '''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    int l252_fi=-1;
    vector<pair<double,double>> l271_failed;   // did not pack, tight -> loose
    bool l271_done=false;
    for (auto& f:frms){
        l252_fi++;
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)){
            if (L252) fprintf(stderr,"L252TRY %d 0 0\\n",l252_fi);
            if (L271) l271_failed.push_back(f);
            continue;
        }
        if (L252) fprintf(stderr,"L252TRY %d 1 0\\n",l252_fi);
        // L271. The frames that just failed ARE the density prize: tighter than
        // anything the shipped order reaches, and reachable by the density order.
        // Retry the tightest of them with this layout scaled in as a wire guide --
        // so the density order starts with the wire information that reordering
        // would otherwise have destroyed. The failed packs were already paid for.
        // Modes 5/6 retry the SAME frame instead of a tighter failed one. The
        // point: area_gap is the BOUNDING BOX of the layout, not the frame -- the
        // frame is only an upper bound -- so a denser pack inside the very same
        // frame still buys area. And the antecedent is then never empty, which is
        // what caps modes 1/2/4 at ~6% of profile-runs (56% have no failed frame).
        bool l271_same = (L271==5 || L271==6);
        if (L271 && !l271_done && (l271_same || !l271_failed.empty())){
            l271_done=true;
            int K;
            if (l271_same) K=1;
            else { K=(int)l271_failed.size(); if (K>L271_K) K=L271_K; }
            for (int q=0;q<K;q++){
                pair<double,double> ff = l271_same ? f : l271_failed[q];
                vector<XYWH> guide=c1;
                double sx=ff.first/max(f.first,1e-12), sy=ff.second/max(f.second,1e-12);
                for (int b=0;b<N;b++){
                    // frame-relative, exactly like the L137 hint, then clamped so a
                    // guide position is always a legal point of the smaller frame.
                    guide[b].x=min(max(0.0,guide[b].x*sx), max(0.0,ff.first -guide[b].w));
                    guide[b].y=min(max(0.0,guide[b].y*sy), max(0.0,ff.second-guide[b].h));
                }
                items = (L271==2) ? items_base : items_dens;
                vector<XYWH> c3;
                if (!run_frame(ff.first,ff.second,(L271!=4 && L271!=6),guide,c3)){
                    if (L252) fprintf(stderr,"L271MISS %d\\n",q);
                    continue;
                }
                double sc3=layout_score(c3);
                if (REFINE && L271_REF){
                    vector<XYWH> gd=c3;
                    for (int r=0;r<REFINE_ITERS;r++){
                        vector<XYWH> c4;
                        if (!run_frame(ff.first,ff.second,true,gd,c4)) break;
                        double s4=layout_score(c4);
                        if (s4<sc3){ sc3=s4; c3=c4; }
                        gd.swap(c4);
                    }
                }
                if (L252) fprintf(stderr,"L271HIT %d %.17g %.17g\\n",q,sc3,layout_score(c1));
                if (!have_best||sc3<best_score){ best_score=sc3; best=c3; have_best=true; }
                trials++;
            }
            items=items_base;
        }'''),
    # 8 the deterministic attempt count
    ('        fprintf(stderr,"METRICS %.6f %.6f %d %d %d %d\\n",area,hpwl,vbd,vcl,vmb,nsoft);',
     '        if (L252) fprintf(stderr,"L267PACKS %ld\\n",L271_PACKS);\n'
     '        fprintf(stderr,"METRICS %.6f %.6f %d %d %d %d\\n",area,hpwl,vbd,vcl,vmb,nsoft);'),
]


# ── L272: stop skipping unplaced neighbours ──────────────────────────────────
# The wire term's fallback chain is  placed -> use_prev guide -> NOTHING. That
# last step is the whole reason reordering is expensive: an item committed early
# has few placed neighbours, so most of its wire is invisible and it is scored on
# `area` alone. L271 repairs it with a real layout, which costs a pack. This
# repairs it with the L137 GORDIAN hint, which is already in the input, is
# frame-relative, is ORDER-INDEPENDENT, and costs nothing at all.
#
# Priority stays placed > guide > hint: an actual position always beats an
# estimate. If the hint is absent (it is cores-gated >=40, so it IS present on
# the 48c grader but NOT on a small dev box) the clause is skipped and the
# mechanism silently reduces to shipped -- which is why the liveness check must
# run at the deployment core count.
_WIRE_PREV = ("else if (use_prev){ ncx=prev_pos[nb.first].x+prev_pos[nb.first].w/2; "
              "ncy=prev_pos[nb.first].y+prev_pos[nb.first].h/2; }")
_WIRE_HINT = ("else if (L272 && (int)hint.size()==N){ "
              "ncx=hint[nb.first].first*fw; ncy=hint[nb.first].second*fh; }")
_WIRE_SITES = 5      # 3 in the anchored first-pass, 1 in GUIDE_MED, 1 in the singles loop


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
    n = out.count(_WIRE_PREV)
    if n != _WIRE_SITES:
        print("!! wire fallback appears {} times, expected {}".format(n, _WIRE_SITES))
        return 1
    out = out.replace(_WIRE_PREV, _WIRE_PREV + "\n                        " + _WIRE_HINT)
    DST.write_bytes(out.encode("utf-8"))
    print("wrote {}  ({} patches + {} wire sites)".format(
        DST.name, len(PATCHES), _WIRE_SITES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
