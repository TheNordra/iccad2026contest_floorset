"""L256 -- shrink the winning frame and repair by ruin-and-recreate.

The mechanism L252-L255 sized. The greedy jams ~3.4pp below its own frame's
allowance with >=10x the needed area free (L254) because it places once and never
relocates; L255 priced perfect relocation of the last 2% at +1.22% of quality and
showed the prize grows to ~+3.2% if ~10% of the design is re-placed.

So: after the frame trial loop has picked a winner, shrink that frame a step,
tear out the items that no longer fit plus a budget of their neighbours toward the
far corner, and let the SHIPPED candidate machinery re-place them. Accept only if
layout_score improves; stop at the first refusal.

Design notes that matter:

  * Ruin is at ITEM granularity, never block. pack_in_frame's main loop skips any
    item that is PARTIALLY placed ("partial: leave to other items/frames"), so a
    block-level keep mask would silently leave holes and still return true.
  * Items containing a preplaced member are never ruined. Their movable members
    are positioned by the anchored first-pass, not the item loop, and the
    preplaced blocks are re-seeded separately every call.
  * The keep set is seeded exactly the way preplaced blocks are -- same bounds
    check, same overlap check, same grid/bbox bookkeeping -- so a seed that does
    not fit the smaller frame fails the pack rather than corrupting it.
  * Runs BEFORE compaction, so the shipped post-processing still applies.

Everything is behind ICCAD_L256; default off is bit-identical to the shipped
binary (proved by l252_identity.py, not asserted).

    ICCAD_L256=1          enable
    ICCAD_L256_STEP=0.99  linear shrink per iteration (area = STEP^2)
    ICCAD_L256_RUIN=0.12  fraction of total block area to tear out
    ICCAD_L256_ITERS=6    max shrink steps
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l256.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

PATCHES = [
    # 1 ---- flags -----------------------------------------------------------
    ('static bool FRAME_REPORT = false;',
     'static bool   L256 = false;         // ICCAD_L256=1: shrink + ruin-and-recreate\n'
     'static double L256_STEP = 0.99;     // linear shrink per iteration\n'
     'static double L256_RUIN = 0.12;     // fraction of total area to tear out\n'
     'static int    L256_ITERS = 6;\n'
     'static bool   L256_SEED = false;    // pack_in_frame: seed the keep set\n'
     'static bool   L256_DBG = false;     // ICCAD_L256_DBG=1: why did it stop?\n'
     'static int    L256_MODE = 1;        // 0=seed kept blocks, 1=guided re-pack\n'
     'static vector<char> L256_KEEP;\n'
     'static vector<XYWH> L256_POS;\n'
     'static bool FRAME_REPORT = false;'),
    # 2 ---- env -------------------------------------------------------------
    ('if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;',
     'if (getenv("ICCAD_L256")) L256=true;\n'
     '    if (const char* e=getenv("ICCAD_L256_STEP")) L256_STEP=atof(e);\n'
     '    if (const char* e=getenv("ICCAD_L256_RUIN")) L256_RUIN=atof(e);\n'
     '    if (const char* e=getenv("ICCAD_L256_ITERS")) L256_ITERS=atoi(e);\n'
     '    if (getenv("ICCAD_L256_DBG")) L256_DBG=true;\n'
     '    if (const char* e=getenv("ICCAD_L256_MODE")) L256_MODE=atoi(e);\n'
     '    if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;'),
    # 3 ---- seed the keep set, exactly like preplaced ------------------------
    (r'''        rects.push_back(pos[i]); g46.add((int)rects.size()-1); bbox_add(pos[i].x,pos[i].y,pos[i].w,pos[i].h); done[i]=1;
    }''',
     r'''        rects.push_back(pos[i]); g46.add((int)rects.size()-1); bbox_add(pos[i].x,pos[i].y,pos[i].w,pos[i].h); done[i]=1;
    }
    // L256: blocks the caller wants preserved enter on the SAME path preplaced
    // blocks do -- bounds, overlap, grid, bbox -- so an infeasible seed fails the
    // pack instead of corrupting it.
    if (L256_SEED){
        for (int i=0;i<N;i++){
            if (done[i] || !L256_KEEP[i]) continue;
            const XYWH& q=L256_POS[i];
            if (q.x<-TOL||q.y<-TOL||q.x+q.w>fw+TOL||q.y+q.h>fh+TOL) return false;
            for (const XYWH&r:rects) if (rect_overlap(q.x,q.y,q.w,q.h,r.x,r.y,r.w,r.h)) return false;
            out[i]=q; rects.push_back(q); g46.add((int)rects.size()-1);
            bbox_add(q.x,q.y,q.w,q.h); done[i]=1;
        }
    }'''),
    # 4 ---- remember which frame won ----------------------------------------
    (r'''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frms){''',
     r'''    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    double best_fw=0, best_fh=0;
    for (auto& f:frms){'''),
    (r'''        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true; }
        if (trials>=max_trials) break;''',
     r'''        if (!have_best||sc<best_score){ best_score=sc; best=c1; have_best=true;
                                        best_fw=f.first; best_fh=f.second; }
        if (trials>=max_trials) break;'''),
    # 5 ---- the shrink loop, before compaction -------------------------------
    (r'''    if (!have_best) best=shelf_fallback(order);''',
     r'''    if (!have_best) best=shelf_fallback(order);
    // ─── L256: shrink + ruin-and-recreate ────────────────────────────────────
    if (L256 && have_best && best_fw>0){
        if (L256_DBG) fprintf(stderr,"L256DBG enter N=%d best_fw=%.6g best_fh=%.6g csc=%.10g\n",N,best_fw,best_fh,csc_of(best,compute_nsoft()));
        double tot=0; for (int i=0;i<N;i++) tot+=dims[i].first*dims[i].second;
        double l6_nsoft=compute_nsoft();
        double base_csc=csc_of(best,l6_nsoft);
        double cfw=best_fw, cfh=best_fh, cur_score=base_csc;
        vector<XYWH> cur=best;
        const size_t NI=items_base.size();
        vector<char> pre(NI,0);
        for (size_t k=0;k<NI;k++)
            for (int b:items_base[k].blocks) if (placed[b]) pre[k]=1;
        // the SAME floors frame_candidates() applies. Without them the shrink can
        // demand a frame narrower than the widest block: the pack fails instantly
        // and reads as "the packer is at its limit" when it is arithmetic.
        double l6_iw=1,l6_ih=1,l6_pw=0,l6_ph=0;
        for (int i=0;i<N;i++){
            l6_iw=max(l6_iw,dims[i].first); l6_ih=max(l6_ih,dims[i].second);
            if (placed[i]){ l6_pw=max(l6_pw,pos[i].x+pos[i].w); l6_ph=max(l6_ph,pos[i].y+pos[i].h); }
        }
        double l6_fw=max(l6_pw,l6_iw)+FRAME_EPS, l6_fh=max(l6_ph,l6_ih)+FRAME_EPS;
        if (L256_DBG) fprintf(stderr,"L256DBG floors fw>=%.6g fh>=%.6g (have %.6g x %.6g)\n",l6_fw,l6_fh,cfw,cfh);
        for (int step=0; step<L256_ITERS; step++){
            double nfw=max(cfw*L256_STEP,l6_fw), nfh=max(cfh*L256_STEP,l6_fh);
            if (!(nfw<cfw-TOL||nfh<cfh-TOL)){
                if (L256_DBG) fprintf(stderr,"L256DBG stop: clamped, no shrink left\n");
                break;
            }
            vector<char> ruin(NI,0); double ruined=0;
            // (a) everything that no longer fits
            for (size_t k=0;k<NI;k++){
                if (pre[k]) continue;
                bool over=false;
                for (int b:items_base[k].blocks){
                    const XYWH& q=cur[b];
                    if (q.x+q.w>nfw+TOL||q.y+q.h>nfh+TOL){ over=true; break; }
                }
                if (over){
                    ruin[k]=1;
                    for (int b:items_base[k].blocks) ruined+=dims[b].first*dims[b].second;
                }
            }
            // (b) grow toward the far corner until the ruin budget is met
            vector<pair<double,size_t>> ord2;
            for (size_t k=0;k<NI;k++){
                if (ruin[k]||pre[k]) continue;
                double cx=0,cy=0; int cnt=0;
                for (int b:items_base[k].blocks){
                    cx+=cur[b].x+cur[b].w/2; cy+=cur[b].y+cur[b].h/2; cnt++;
                }
                if (!cnt) continue;
                ord2.push_back({-((cx/cnt)/max(nfw,1e-9)+(cy/cnt)/max(nfh,1e-9)),k});
            }
            sort(ord2.begin(),ord2.end());
            for (auto& pr:ord2){
                if (ruined >= L256_RUIN*tot) break;
                ruin[pr.second]=1;
                for (int b:items_base[pr.second].blocks)
                    ruined+=dims[b].first*dims[b].second;
            }
            int nruin=0; for (size_t k=0;k<NI;k++) if (ruin[k]) nruin++;
            if (L256_DBG) fprintf(stderr,"L256DBG step=%d items=%d ruin=%d frac=%.4f fw=%.6g->%.6g\n",
                                  step,(int)NI,nruin,ruined/max(tot,1e-18),cfw,nfw);
            if (!nruin){ if (L256_DBG) fprintf(stderr,"L256DBG stop: nothing to ruin\n"); break; }
            // (c) keep everything else exactly where it is, and re-pack the rest
            L256_KEEP.assign(N,0); L256_POS.assign(N,XYWH{0,0,0,0});
            for (size_t k=0;k<NI;k++) if (!ruin[k])
                for (int b:items_base[k].blocks){ L256_KEEP[b]=1; L256_POS[b]=cur[b]; }
            items=items_base;
            vector<XYWH> cand, dummy2;
            bool ok2;
            if (L256_MODE==0){
                L256_SEED=true;
                ok2=run_frame(nfw,nfh,false,dummy2,cand);
                L256_SEED=false;
            } else {
                ok2=run_frame(nfw,nfh,true,cur,cand);
            }
            if (!ok2){ if (L256_DBG) fprintf(stderr,"L256DBG stop: repack FAILED\n"); break; }
            double s2=csc_of(cand,l6_nsoft);
            if (L256_DBG) fprintf(stderr,"L256DBG   repack ok  s2=%.10g cur=%.10g  %s\n",
                                  s2,cur_score,(s2<cur_score)?"ACCEPT":"reject");
            if (!(s2<cur_score)) break;
            cur=cand; cur_score=s2; cfw=nfw; cfh=nfh;
        }
        if (cur_score<base_csc) best=cur;   // best_score is dead after this point
    }'''),
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
