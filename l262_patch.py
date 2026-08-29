"""L262 -- eviction inside pack_in_frame. The move L259/L260/L261 identified.

L259: at the jam the largest unplaced block has ZERO legal positions, so no
recreate can finish given the greedy's prefix. L260: displacing exactly ONE
placed block opens a slot (8/8 cases, median 0.75% of the design). L261: running
the cascade completes the whole layout in 7/8 cases at a median of 2 evictions.

So: when an item finds no origin, instead of failing the frame, evict the
minimum-count set of already-placed items blocking the cheapest anchor, and
RE-QUEUE both the item and the evicted ones.

🔑 The item is deliberately NOT placed at the eviction anchor. It goes back on
the worklist, so its real placement returns through item_candidates() + the
shipped scoring, and the wire / boundary / anchor terms are all still respected.
Eviction only creates the opportunity; it never chooses the position. That is the
difference from L256, whose recreate was the same greedy that had just jammed.

Never evictable: preplaced blocks, and movable members placed by the anchored
first-pass (they are attached to preplaced walls). Those carry owner -1.

Eviction physically erases the rects, so the M46 grid and the running bbox are
rebuilt from the survivors -- both are add-only structures and a stale one would
silently corrupt every subsequent overlap test.

    ICCAD_L262=1        enable
    ICCAD_L262_MAX=N    total eviction budget per pack (default 24)
"""
import hashlib
import sys
from pathlib import Path

DIR = Path(__file__).parent
SRC = DIR / "constructive.cpp"
DST = DIR / "constructive_l262.cpp"
EXPECT_SRC_MD5 = "e2c7b2f418ef2b70b6bff99f7adfbd37"

PATCHES = [
    # 1 flags
    ('static bool FRAME_REPORT = false;',
     'static bool L252 = false;        // ICCAD_L252=1: emit the frame ladder\n'
     'static bool L262 = false;        // ICCAD_L262=1: evict-and-requeue on a jam\n'
     'static int  L262_MAX = 24;       // total eviction budget per pack\n'
     'static bool FRAME_REPORT = false;'),
    ('if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;',
     'if (getenv("ICCAD_L252")) L252=true;\n'
     '    if (getenv("ICCAD_L262")) L262=true;\n'
     '    if (const char* e=getenv("ICCAD_L262_MAX")) L262_MAX=atoi(e);\n'
     '    if (getenv("ICCAD_FRAME_REPORT")) FRAME_REPORT=true;'),
    (r"""    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    return frames;""",
     r"""    sort(frames.begin(),frames.end(),[](const pair<double,double>&A,const pair<double,double>&B){
        double aa=A.first*A.second,ab=B.first*B.second;
        if (fabs(aa-ab)>TOL) return aa<ab; return max(A.first,A.second)<max(B.first,B.second);
    });
    if (L252){
        fprintf(stderr,"L252TOT %.17g\n",total);
        for (size_t i=0;i<frames.size();i++)
            fprintf(stderr,"L252FRM %d %.17g %.17g\n",(int)i,frames[i].first,frames[i].second);
    }
    return frames;"""),
    (r"""    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    for (auto& f:frms){
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)) continue;""",
     r"""    vector<XYWH> best; bool have_best=false; double best_score=1e300; int trials=0;
    int l252_fi=-1;
    for (auto& f:frms){
        l252_fi++;
        items=items_base;
        vector<XYWH> c1, dummy;
        if (!run_frame(f.first,f.second,false,dummy,c1)){
            if (L252) fprintf(stderr,"L252TRY %d 0 0\n",l252_fi);
            continue;
        }
        if (L252) fprintf(stderr,"L252TRY %d 1 0\n",l252_fi);"""),
    # 2 rect ownership vector
    ('    out=pos; vector<XYWH> rects; bbox_reset();',
     '    out=pos; vector<XYWH> rects; bbox_reset();\n'
     '    // L262: owner item index per rect; -1 = never evictable (preplaced, and\n'
     '    // anchored first-pass members attached to preplaced walls).\n'
     '    vector<int> rect_own;'),
    # 3 preplaced -> owner -1
    (r'''        rects.push_back(pos[i]); g46.add((int)rects.size()-1); bbox_add(pos[i].x,pos[i].y,pos[i].w,pos[i].h); done[i]=1;''',
     r'''        rects.push_back(pos[i]); rect_own.push_back(-1); g46.add((int)rects.size()-1); bbox_add(pos[i].x,pos[i].y,pos[i].w,pos[i].h); done[i]=1;'''),
    # 4 anchored first-pass -> owner -1
    (r'''                out[b]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1);
                bbox_add(bx,by,bw,bh); done[b]=1; cluster_rects.push_back({bx,by,bw,bh});''',
     r'''                out[b]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); rect_own.push_back(-1); g46.add((int)rects.size()-1);
                bbox_add(bx,by,bw,bh); done[b]=1; cluster_rects.push_back({bx,by,bw,bh});'''),
    # 5 the worklist loop
    (r'''    for (const Item& it:items){
        bool all_done=true; for (int b:it.blocks) if(!done[b]){ all_done=false; break; }''',
     r'''    vector<int> l262_work; l262_work.reserve(items.size()*2);
    for (size_t _k=0;_k<items.size();_k++) l262_work.push_back((int)_k);
    vector<int> l262_try(items.size(),0);
    int l262_ev=0;
    for (size_t _wp=0; _wp<l262_work.size(); _wp++){
        const int _ii=l262_work[_wp];
        const Item& it=items[_ii];
        bool all_done=true; for (int b:it.blocks) if(!done[b]){ all_done=false; break; }'''),
    # 6 free-aspect single: fall through to the generic path when L262 is on
    (r'''                if (!found) return false;
                out[sb]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); g46.add((int)rects.size()-1); bbox_add(bx,by,bw,bh); done[sb]=1;
                continue;''',
     r'''                if (!found && !L262) return false;
                if (found){
                out[sb]={bx,by,bw,bh}; rects.push_back({bx,by,bw,bh}); rect_own.push_back(_ii); g46.add((int)rects.size()-1); bbox_add(bx,by,bw,bh); done[sb]=1;
                continue;
                }'''),
    # 7 the eviction itself
    (r'''        if (!found) return false;
        for (size_t k=0;k<it.blocks.size();k++){
            int b=it.blocks[k]; double rx=bx+it.offs[k].first, ry=by+it.offs[k].second;
            double bw=dims[b].first, bh=dims[b].second;
            out[b]={rx,ry,bw,bh}; rects.push_back({rx,ry,bw,bh}); g46.add((int)rects.size()-1); bbox_add(rx,ry,bw,bh); done[b]=1;
        }''',
     r'''        if (!found){
            if (!L262 || l262_ev>=L262_MAX || l262_try[_ii]>=2) return false;
            // Cheapest anchor by NUMBER of evictable items displaced, tie-broken by
            // their area. An anchor overlapping any owner -1 rect is disqualified.
            double iw=it.w, ih=it.h;
            vector<double> xs, ys; xs.push_back(0.0); ys.push_back(0.0);
            for (const XYWH&r:rects){
                double a=r.x+r.w, b2=r.x-iw, c2=r.y+r.h, d2=r.y-ih;
                if (a>=-TOL && a+iw<=fw+TOL) xs.push_back(a);
                if (b2>=-TOL && b2+iw<=fw+TOL) xs.push_back(b2);
                if (c2>=-TOL && c2+ih<=fh+TOL) ys.push_back(c2);
                if (d2>=-TOL && d2+ih<=fh+TOL) ys.push_back(d2);
            }
            sort(xs.begin(),xs.end()); xs.erase(unique(xs.begin(),xs.end()),xs.end());
            sort(ys.begin(),ys.end()); ys.erase(unique(ys.begin(),ys.end()),ys.end());
            int bestc=1<<30; double besta=1e300; set<int> bestv;
            for (double x:xs) for (double y:ys){
                if (x<-TOL||y<-TOL||x+iw>fw+TOL||y+ih>fh+TOL) continue;
                set<int> vic; double var=0; bool bad=false;
                for (size_t r=0;r<rects.size();r++){
                    if (!rect_overlap(x,y,iw,ih,rects[r].x,rects[r].y,rects[r].w,rects[r].h)) continue;
                    if (rect_own[r]<0){ bad=true; break; }
                    vic.insert(rect_own[r]); var+=rects[r].w*rects[r].h;
                }
                if (bad || vic.empty()) continue;
                int c=(int)vic.size();
                if (c<bestc || (c==bestc && var<besta)){ bestc=c; besta=var; bestv=vic; }
            }
            if (bestv.empty()) return false;
            for (int v:bestv) for (int b:items[v].blocks) done[b]=0;
            vector<XYWH> _nr; vector<int> _no;
            for (size_t r=0;r<rects.size();r++)
                if (rect_own[r]<0 || !bestv.count(rect_own[r])){
                    _nr.push_back(rects[r]); _no.push_back(rect_own[r]);
                }
            rects.swap(_nr); rect_own.swap(_no);
            // g46 and the running bbox are ADD-ONLY: rebuild both from the
            // survivors, or every later overlap test is against a stale index.
            g46.init(fw,fh,&rects); bbox_reset();
            for (size_t r=0;r<rects.size();r++){
                g46.add((int)r); bbox_add(rects[r].x,rects[r].y,rects[r].w,rects[r].h);
            }
            l262_try[_ii]++;
            l262_work.push_back(_ii);
            for (int v:bestv){ l262_work.push_back(v); l262_ev++; }
            continue;
        }
        for (size_t k=0;k<it.blocks.size();k++){
            int b=it.blocks[k]; double rx=bx+it.offs[k].first, ry=by+it.offs[k].second;
            double bw=dims[b].first, bh=dims[b].second;
            out[b]={rx,ry,bw,bh}; rects.push_back({rx,ry,bw,bh}); rect_own.push_back(_ii); g46.add((int)rects.size()-1); bbox_add(rx,ry,bw,bh); done[b]=1;
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
