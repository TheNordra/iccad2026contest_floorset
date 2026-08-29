"""L267/L268 -- one probe binary, two independent mechanisms.

L267  ADAPTIVE FRAME SEARCH (handoff 2026-08-30 2.1).
      Every frame ladder so far was FIXED, which forces a choice L264/L266 proved
      unwinnable: dense rungs sit at the per-case cliff but every rung below s_min
      is a full failed pack (max-setter x1.48), and a short ladder is affordable
      but its rungs are constants fitted to one sample (0% transfer, twice).
      The trial loop already learns, per case, which frames fail -- and throws it
      away. This searches for the cliff instead of guessing it:
         phase 1  walk the profile's OWN scales upward at ONE aspect until a pack
                  succeeds  -> bracket (lo fails, hi packs)
         phase 2  bisect inside the bracket with what is left of the probe budget
         phase 3  emit RUNGS rungs upward from the located cliff x every aspect,
                  clamped and area-sorted exactly as frame_candidates() does
      Probes are bare pack_in_frame calls -- no refinement, no nudges, no scoring;
      the same unit of work a failing rung costs today, capped at L267_PROBES.
      No fitted constants: every rung position is derived per case.

L268  BIG-FIRST COMMITMENT ORDER (handoff 3.4).
      L259: at the jam the LARGEST unplaced block has ZERO legal positions while
      smaller ones have thousands; L261: one eviction completes the layout. So the
      defect is the order of commitment, not the search. The shipped sort keys on
      bscore CLASS first (boundary items, whatever their size), area only inside a
      class. 1 = global area descending, 2 = global max-dimension descending.

Both default off => bit-identical to the shipped binary.

    ICCAD_L252=1          the L252 stderr emitters (ladder + per-trial ok) and
                          L267BIS / L267SEL / L267PACKS.  stderr-only.
    ICCAD_L267=1          adaptive frame search
    ICCAD_L267_PROBES=K   probe budget for phases 1+2   (default 5)
    ICCAD_L267_STEP=x     rung step above the cliff; <=0 => use the bracket width
    ICCAD_L267_RUNGS=R    rungs emitted                 (default 4)
    ICCAD_L268=1|2        big-first commitment order
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l267.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

ADAPTIVE = r'''
// --- L267: adaptive frame search ---------------------------------------------
// See l267_patch.py. Replaces frame_candidates() when ICCAD_L267=1; falls back to
// it verbatim if nothing in the profile's own ladder packs.
static vector<pair<double,double>> l267_frames(const vector<Item>& its) {
    double total=0,max_iw=1,max_ih=1,pre_w=0,pre_h=0;
    for (int i=0;i<N;i++){
        total+=dims[i].first*dims[i].second;
        max_iw=max(max_iw,dims[i].first); max_ih=max(max_ih,dims[i].second);
        if (placed[i]){ pre_w=max(pre_w,pos[i].x+pos[i].w); pre_h=max(pre_h,pos[i].y+pos[i].h); }
    }
    double base=sqrt(max(total,1.0));
    vector<double> aspects = FRAME_ASPECTS.empty() ? vector<double>{1.0,1.35,0.75,1.8,0.55}
                                                   : FRAME_ASPECTS;
    vector<double> scales = !FRAME_SCALES.empty() ? FRAME_SCALES
        : (N>=60&&N<80)?vector<double>{1.00,1.03,1.05,1.15,1.35,1.65,2.10}
                       :vector<double>{1.05,1.15,1.35,1.65,2.10};
    sort(scales.begin(),scales.end());
    if (aspects.empty()||scales.empty()) return frame_candidates();
    double W0=max(pre_w,max_iw)+FRAME_EPS, H0=max(pre_h,max_ih)+FRAME_EPS;
    // !! THE CLAMP. frame_candidates() floors every frame at max(pre,max_i)+EPS.
    // Anything constructing a frame outside that function must re-apply it, or it
    // demands a frame narrower than the widest block and fails instantly while
    // looking like a packer limit (this cost L256 an hour).
    auto mk=[&](double s,double a)->pair<double,double>{
        double r=sqrt(a);
        return make_pair( max(base*s*r, W0), max(base*s/r, H0) );
    };
    // The tightest frame ANY aspect can produce is both dimensions on their floors,
    // (W0,H0) -- every aspect clamps to it once s is small enough. t0 is its
    // EFFECTIVE scale, sqrt(area/SumA), and it is a hard geometric floor. This is
    // also why 13/40 of L252's cases had no failing rung at all.
    double t0 = sqrt(max(W0*H0,1e-18)/max(total,1e-18));
    // The probe frame at scale s is the SMALLEST the aspect list can make at that
    // scale, so its area is total*s^2 whenever the floors allow it at all -- i.e.
    // nominal scale == effective scale, which is what makes the located value a
    // usable ladder anchor. Bisecting a clamped aspect instead searches a variable
    // that is flat over most of its range.
    auto mkbest=[&](double s)->pair<double,double>{
        pair<double,double> bf=mk(s,aspects[0]); double ba=bf.first*bf.second;
        for (size_t i=1;i<aspects.size();i++){
            pair<double,double> f=mk(s,aspects[i]); double ar=f.first*f.second;
            if (ar<ba-TOL){ ba=ar; bf=f; }
        }
        return bf;
    };
    int used=0;
    auto probe=[&](double s)->bool{
        pair<double,double> f=mkbest(s);
        vector<XYWH> tmp; bool sv=use_prev; use_prev=false;
        bool ok=pack_in_frame(f.first,f.second,its,tmp);
        use_prev=sv; used++;
        if (L252) fprintf(stderr,"L267BIS %.6f %d\n",s,ok?1:0);
        return ok;
    };
    double lo=t0, hi=-1; bool any_above=false;
    for (size_t si=0; si<scales.size(); si++){
        double s=scales[si];
        if (s<=t0+1e-12) continue;               // at or below the geometric floor
        any_above=true;
        if (used>=L267_PROBES) break;
        if (probe(s)){ hi=s; break; }
        lo=s;
    }
    if (hi<0){
        // The whole ladder sits at or below the floor: the floor frame is then the
        // only thing to test, and if it packs it IS the tightest reachable frame.
        if (!any_above && used<L267_PROBES && probe(t0)) hi=t0;
        else return frame_candidates();          // nothing packed -> shipped set
    }
    while (used<L267_PROBES && hi-lo>1e-4){
        double m=0.5*(lo+hi);
        pair<double,double> fm=mkbest(m), fh=mkbest(hi);
        // Same frame as the one already known to pack => the answer is free. This
        // is the clamped region, where the frame stops changing with s and a probe
        // would buy no information.
        if (fabs(fm.first-fh.first)<TOL && fabs(fm.second-fh.second)<TOL){ hi=m; continue; }
        if (probe(m)) hi=m; else lo=m;
    }
    double step = (L267_STEP>0.0) ? L267_STEP : max(hi-lo,1e-3);
    set<pair<long long,long long>> seen; vector<pair<double,double>> frames;
    for (int k=0;k<L267_RUNGS;k++){
        double s=hi+k*step;
        for (size_t ai=0; ai<aspects.size(); ai++){
            pair<double,double> f=mk(s,aspects[ai]);
            pair<long long,long long> key=make_pair((long long)llround(f.first*1e6),
                                                    (long long)llround(f.second*1e6));
            if (seen.insert(key).second) frames.push_back(f);
        }
    }
    if (frames.empty()) return frame_candidates();
    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    if (L252){
        pair<double,double> fh=mkbest(hi);
        fprintf(stderr,"L267SEL %.6f %.6f %d %.6f %.6f\n",hi,lo,used,
                sqrt(max(fh.first*fh.second,1e-18)/max(total,1e-18)), t0);
        fprintf(stderr,"L252TOT %.17g\n",total);
        for (size_t i=0;i<frames.size();i++)
            fprintf(stderr,"L252FRM %d %.17g %.17g\n",(int)i,frames[i].first,frames[i].second);
    }
    return frames;
}

'''

_SOLVE_HDR = "static void solve() {"

PATCHES = [
    # 1 flags
    ('static bool FRAME_REPORT = false;',
     'static bool L252 = false;        // ICCAD_L252=1: stderr instrumentation only\n'
     'static bool L267 = false;        // ICCAD_L267=1: adaptive frame search\n'
     'static int  L267_PROBES = 5;     // bisection probe budget (packs)\n'
     'static double L267_STEP = 0.01;  // rung step above the cliff; <=0 => bracket width\n'
     'static int  L267_RUNGS = 4;      // rungs emitted by the adaptive ladder\n'
     'static int  L268 = 0;            // ICCAD_L268: see the ordering block below\n'
     'static int  L268_K = 1;          // ICCAD_L268_K: how many items mode 3 hoists\n'
     'static long L267_PACKS = 0;      // deterministic cost counter (pack_in_frame calls)\n'
     'static int  L269 = 0;            // ICCAD_L269: 1 = in-loop bisection, 2 = start loosest\n'
     'static int  L269_PROBES = 5;     // bisection proposals per pipeline\n'
     'static double L269_TOTAL = 0;    // SumA, for scale <-> frame conversion\n'
     'static double L269_W0 = 0, L269_H0 = 0;   // the frame_candidates() clamp floors\n'
     'static bool FRAME_REPORT = false;'),
    # 2 env
    ('    if (getenv("ICCAD_GUIDE_MED")) GUIDE_MED=true;',
     '    if (getenv("ICCAD_L252")) L252=true;\n'
     '    if (getenv("ICCAD_L267")) L267=true;\n'
     '    if (const char* e=getenv("ICCAD_L267_PROBES")){ int v=atoi(e); if (v>0) L267_PROBES=v; }\n'
     '    if (const char* e=getenv("ICCAD_L267_STEP")){ L267_STEP=atof(e); }\n'
     '    if (const char* e=getenv("ICCAD_L267_RUNGS")){ int v=atoi(e); if (v>0) L267_RUNGS=v; }\n'
     '    if (const char* e=getenv("ICCAD_L268")){ int v=atoi(e); if (v>=1&&v<=6) L268=v; }\n'
     '    if (const char* e=getenv("ICCAD_L268_K")){ int v=atoi(e); if (v>0) L268_K=v; }\n'
     '    if (const char* e=getenv("ICCAD_L269")){ int v=atoi(e); if (v==1||v==2) L269=v; }\n'
     '    if (const char* e=getenv("ICCAD_L269_PROBES")){ int v=atoi(e); if (v>0) L269_PROBES=v; }\n'
     '    if (getenv("ICCAD_GUIDE_MED")) GUIDE_MED=true;'),
    # 3 pack counter -- a deterministic cost signal. A wall-clock gate is run-to-run
    #   non-deterministic and breaks every byte-identity comparison (L158).
    ('static bool pack_in_frame(double fw,double fh,const vector<Item>& items,vector<XYWH>& out){\n'
     '    out=pos; vector<XYWH> rects; bbox_reset();',
     'static bool pack_in_frame(double fw,double fh,const vector<Item>& items,vector<XYWH>& out){\n'
     '    L267_PACKS++;\n'
     '    out=pos; vector<XYWH> rects; bbox_reset();'),
    # 4 the L252 emitters in the stock frame_candidates()
    ('''    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    return frames;''',
     '''    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    if (L252 && !L267){
        fprintf(stderr,"L252TOT %.17g\\n",total);
        for (size_t i=0;i<frames.size();i++)
            fprintf(stderr,"L252FRM %d %.17g %.17g\\n",(int)i,frames[i].first,frames[i].second);
    }
    return frames;'''),
    # 5 the adaptive searcher, right before solve()
    (_SOLVE_HDR, ADAPTIVE + _SOLVE_HDR),
    # 6 big-first commitment order + the adaptive call site
    ('''    vector<int> order; for(int i=0;i<N;i++) if(!blocks[i].is_preplaced) order.push_back(i);

    auto frames=frame_candidates();''',
     '''    // L268: big-first commitment order. The shipped sort keys on bscore CLASS
    // first, so a boundary-constrained sliver is committed before the largest
    // free block -- and L259 measured that at the jam the largest unplaced block
    // has ZERO legal positions while smaller ones have thousands. Overrides the
    // class order entirely; stable, so ties keep the shipped order.
    // 1 = global area desc, 2 = global max-dimension desc. Both DROP bscore, and
    //     bscore is what protects vrel: a boundary item committed late can no
    //     longer reach a frame edge. Measured cost of that: vrel 0.0893 -> 0.0998.
    // 3 = hoist the L268_K largest items to the very front, nothing else moved.
    //     L260 measured that displacing exactly ONE placed block opens a slot in
    //     8/8 cases; hoisting a handful is the mirror of it, and it leaves the
    //     wire-visibility order of the other ~99% of items alone.
    // 4 = keep the boundary CLASS order, drop only the compound-item-first key, so
    //     a large single stops waiting behind small compound items.
    // 6 = parameter-free hoist: a FREE item never waits behind a boundary item
    //     SMALLER than itself. Boundary priority is preserved for everything else.
    if (L268==1 || L268==2){
        stable_sort(items.begin(),items.end(),[](const Item&a,const Item&b){
            if (L268==2) return max(a.w,a.h)>max(b.w,b.h);
            return a.w*a.h > b.w*b.h;
        });
    } else if (L268==3 && !items.empty()){
        vector<int> ix(items.size()); for (size_t i=0;i<ix.size();i++) ix[i]=(int)i;
        stable_sort(ix.begin(),ix.end(),[&](int a,int b){
            return items[a].w*items[a].h > items[b].w*items[b].h; });
        int K=min((int)items.size(), L268_K);
        vector<char> up(items.size(),0);
        vector<Item> ord; ord.reserve(items.size());
        for (int k=0;k<K;k++){ up[ix[k]]=1; ord.push_back(items[ix[k]]); }
        for (size_t i=0;i<items.size();i++) if(!up[i]) ord.push_back(items[i]);
        items.swap(ord);
    } else if (L268==4){
        stable_sort(items.begin(),items.end(),[](const Item&a,const Item&b){
            if (a.bscore!=b.bscore) return a.bscore>b.bscore;
            return a.w*a.h > b.w*b.h;
        });
    } else if (L268==6 && !items.empty()){
        double mb=0;
        for (size_t i=0;i<items.size();i++)
            if (items[i].bscore>0) mb=max(mb, items[i].w*items[i].h);
        vector<Item> up, rest;
        for (size_t i=0;i<items.size();i++){
            if (items[i].bscore==0 && items[i].w*items[i].h>mb+TOL) up.push_back(items[i]);
            else rest.push_back(items[i]);
        }
        stable_sort(up.begin(),up.end(),[](const Item&a,const Item&b){
            return a.w*a.h > b.w*b.h; });
        items.clear();
        items.insert(items.end(), up.begin(), up.end());
        items.insert(items.end(), rest.begin(), rest.end());
    }

    vector<int> order; for(int i=0;i<N;i++) if(!blocks[i].is_preplaced) order.push_back(i);

    // L269 needs SumA and the same clamp floors frame_candidates() uses, to convert
    // a scale into a frame and back. ⚠️ Re-applying the clamp is not optional: a
    // frame narrower than the widest block fails instantly while looking like a
    // packer limit (L256).
    if (L269){
        double t=0,miw=1,mih=1,pw=0,ph=0;
        for (int i=0;i<N;i++){
            t+=dims[i].first*dims[i].second;
            miw=max(miw,dims[i].first); mih=max(mih,dims[i].second);
            if (placed[i]){ pw=max(pw,pos[i].x+pos[i].w); ph=max(ph,pos[i].y+pos[i].h); }
        }
        L269_TOTAL=max(t,1.0); L269_W0=max(pw,miw)+FRAME_EPS; L269_H0=max(ph,mih)+FRAME_EPS;
    }

    auto frames = L267 ? l267_frames(items) : frame_candidates();'''),
    # 7 per-trial ok on stderr + the L269 in-loop bisection
    ('''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frms){
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)) continue;''',
     r'''    // L269: the frame list becomes mutable so the loop can propose its OWN next
    // frame. With L269 off nothing is ever inserted and this is the shipped walk.
    vector<pair<double,double>> frms_mut;
    if (L269) {
        if (L269==2 && !frms.empty()) frms_mut.push_back(frms.back());  // loosest only
        else frms_mut.assign(frms.begin(), frms.end());
    }
    const vector<pair<double,double>>& fl = L269 ? frms_mut : frms;
    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    int l252_fi=-1;
    // L269 bisection state. lo = the loosest scale KNOWN to fail (1.0 is 100%
    // utilisation, unpackable by construction); hi = the tightest scale known to
    // pack, and l269_ra is the aspect it packed at. Anchoring on an aspect that
    // already packed is what makes this MONOTONE: it can never end up looser than
    // the shipped ladder's own first success, which is exactly where L267's
    // single-aspect guess lost (13/40 cases looser than ship).
    double l269_lo=1.0, l269_hi=-1.0, l269_ra=1.0;
    int l269_used=0;
    for (size_t l269_i=0; l269_i<fl.size(); l269_i++){
        pair<double,double> f = fl[l269_i];        // by value: fl may be spliced
        l252_fi++;
        double l269_s = (L269 && L269_TOTAL>0)
            ? sqrt(max(f.first*f.second,1e-18)/L269_TOTAL) : 0.0;
        items=items_base;
        vector<XYWH> c1, dummy;
        bool l269_ok = run_frame(f.first,f.second,false,dummy,c1);
        if (L269){
            if (l269_ok){
                if (l269_hi<0 || l269_s<l269_hi){ l269_hi=l269_s; l269_ra=sqrt(f.first/f.second); }
            } else if (l269_s>l269_lo && (l269_hi<0 || l269_s<l269_hi-1e-9)) l269_lo=l269_s;
            // Propose the next frame by bisection instead of taking the ladder's.
            // A proposal that packs becomes a real trial (work the loop would have
            // spent on another aspect anyway); one that fails costs one pack, and
            // there are at most L269_PROBES of them.
            if (l269_hi>0 && l269_used<L269_PROBES && l269_hi-l269_lo>1e-3){
                double m=0.5*(l269_lo+l269_hi), base=sqrt(L269_TOTAL);
                double w=max(base*m*l269_ra, L269_W0), h=max(base*m/l269_ra, L269_H0);
                if (fabs(w*h - L269_TOTAL*l269_hi*l269_hi)>1e-9){
                    l269_used++;
                    frms_mut.insert(frms_mut.begin()+l269_i+1, make_pair(w,h));
                    if (L252) fprintf(stderr,"L269BIS %.6f\n",m);
                }
            }
        }
        if (!l269_ok){
            if (L252) fprintf(stderr,"L252TRY %d 0 0\n",l252_fi);
            continue;
        }
        if (L252) fprintf(stderr,"L252TRY %d 1 0\n",l252_fi);'''),
    # 8 the deterministic cost readout
    ('        fprintf(stderr,"METRICS %.6f %.6f %d %d %d %d\\n",area,hpwl,vbd,vcl,vmb,nsoft);',
     '        if (L252) fprintf(stderr,"L267PACKS %ld\\n",L267_PACKS);\n'
     '        fprintf(stderr,"METRICS %.6f %.6f %d %d %d %d\\n",area,hpwl,vbd,vcl,vmb,nsoft);'),
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
